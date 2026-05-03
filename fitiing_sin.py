import torch

import math

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
dtype = torch.float64
torch.set_default_dtype(dtype)

def graph(input_x, input_y, predict_x, predict_y, iteration):
    fig, ax = plt.subplots(figsize=(8, 8))

    # 1. 파란색 점 먼저 그리기

    ax.plot(input_x, input_y, color='blue', linewidth=1 ,alpha=0.7, label='sin x')

    # 2. 3차 함수 실선 그리기 (범위 밖으로 나가도 일단 다 그림)

    ax.scatter(predict_x, predict_y, color='red', s=2 , alpha=0.5, label='3rd order fit')

    # 핵심: 그래프 화면을 파란색 점 범위로 '강제 고정' (클리핑)

    # input_x와 input_y의 최소/최대값까지만 보여주게 설정합니다.

    ax.set_xlim(input_x.min(), input_x.max())
    ax.set_ylim(input_y.min(), input_y.max())

    # --- 이하 좌표 평면 설정 (동일) ---
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.legend()
    plt.savefig(f'graph{iteration:04d}.png')
    plt.close(fig)


x = torch.linspace(-math.pi , math.pi, 2000)   
y = torch.sin(x)

y_random = torch.linspace(-1000,10000 , 2000)
y_SE = (y_random - y ) ** 2


x_numpy = x.cpu().numpy()
y_numpy = y.cpu().numpy()


y_random = y_random.cpu().numpy()
y_SE = y_SE.cpu().numpy()



lr=1e-6



a = torch.randn(())
b = torch.randn(())
c = torch.randn(())
d = torch.randn(())

loss_history = []
gradient_history = []


for i in range(5000):
    pred_y = a + b * x + c * x ** 2 + d * x ** 3
    pred_y_numpy = pred_y.cpu().numpy()

    SSE = torch.sum((pred_y - y).pow(2)).item()
    loss_history.append(SSE)


    grad = (pred_y - y) * 2.0

    grad_save =torch.abs(torch.sum(grad))
    grad_sum = torch.sum(grad)
    gradient_history.append(grad_save.item())

    a_grad = (grad * 1).sum()
    b_grad = (grad * x).sum()
    c_grad = (grad * x ** 2).sum()
    d_grad = (grad * x ** 3).sum()

    a -= lr * a_grad
    b -= lr * b_grad
    c -= lr * c_grad
    d -= lr * d_grad

    print(f"{i}번째 SSE 와 grad  :{SSE} , {grad_sum}")


    if i % 10 == 0:
        graph(x_numpy, y_numpy, x_numpy, pred_y_numpy, i)



plt.figure(figsize=(10, 6))
plt.plot(loss_history, color = 'tab:red', linewidth = 2 )

plt.yscale('log')

plt.title("training loss (SSE)",fontsize = 15)
plt.xlabel("iteration",fontsize = 15)
plt.ylabel("loss_value",fontsize = 15)


plt.savefig('loss.png')
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(gradient_history, color = 'tab:red', linewidth = 2 )

plt.yscale('log')

plt.title("gradient",fontsize = 15)
plt.xlabel("iteration",fontsize = 15)
plt.ylabel("gradient_value",fontsize = 15)


plt.savefig('gradient.png')
plt.show()














