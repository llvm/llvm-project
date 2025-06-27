// 定义 FT7004 的调用约定

def RetCC_FT7004 : CallingConv<[
    CCIfType<[i32], CCAssignToReg<[R30, R31]>>,
]>;

def CC_FT7004 : CallingConv<[
    CCIfType<[i32], CCAssignToReg<[
        R20, R21, R22, R23, R24, R25, R26, R27, R28, R29, R30, R31
    ]>>,
    CCAssignToStack<4, 4>
]>;

def CSR_FT7004 : CalleeSavedRegs<(add R0)>;
