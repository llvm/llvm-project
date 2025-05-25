//定义了一个名为RetCC_PE的调用约定。调用约定决定了函数调用时参数如何传递，返回值如何存放等规则。
//CCIfType<[i32]当返回值为i32时，执行后序
//CCAssignToReg将返回值分配到 ​X10​（RISC-V 的 a0 寄存器）和 ​X11​（a1 寄存器）
def RetCC_PE : CallingConv<[
    CCIfType<[i32],CCAssignToReg<[X10,X11]>>,
]>;