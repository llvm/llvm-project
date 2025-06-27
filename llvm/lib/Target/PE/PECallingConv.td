//定义了一个名为RetCC_PE的调用约定。调用约定决定了函数调用时参数如何传递，返回值如何存放等规则。
//CCIfType<[i32]当返回值为i32时，执行后序
//CCAssignToReg将返回值分配到 RS2,RS3(暂用)
def RetCC_PE : CallingConv<[
    CCIfType<[i32],CCAssignToReg<[RS30,RS31]>>,
]>;

//定义传参寄存器

def CC_PE : CallingConv<[
    CCIfType<[i32],CCAssignToReg<[
        RS20,RS21,RS22,RS23,RS24,RS25,RS26,RS27,RS28,RS29,RS30,RS31
    ]>>,
    //表示每个参数占 4 字节，4 字节对齐
    CCAssignToStack<4, 4>
]>;

//寄存器保护，不能随意修改，修改前应该save
def CSR : CalleeSavedRegs<(add
    RS0
)>;