// FT7004 寄存器定义

class FT7004Reg<string n> : Register<n> {
    let Namespace = "FT7004";
}

// 示例寄存器定义
def R0  : FT7004Reg<"R0">,  DwarfRegNum<[0]>;
def R1  : FT7004Reg<"R1">,  DwarfRegNum<[1]>;
def R2  : FT7004Reg<"R2">,  DwarfRegNum<[2]>;
def R3  : FT7004Reg<"R3">,  DwarfRegNum<[3]>;
def R4  : FT7004Reg<"R4">,  DwarfRegNum<[4]>;
def R5  : FT7004Reg<"R5">,  DwarfRegNum<[5]>;
def R6  : FT7004Reg<"R6">,  DwarfRegNum<[6]>;
def R7  : FT7004Reg<"R7">,  DwarfRegNum<[7]>;
// ... 可继续补充更多寄存器 ...

// 寄存器类定义
// def GPR : RegisterClass<"FT7004", [i32], 32, (add R0, R1, R2, R3, R4, R5, R6, R7)>;
