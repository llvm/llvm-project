// 包含必要的 LLVM 定义文件
include "llvm/IR/Intrinsics.td"
include "llvm/Target/Target.td"
// 定义 PE 指令集信息
def PEInstrInfo : InstrInfo {
  let Name = "PE";         // 指令集名称
  let AllowRegisterRenaming = 0;  // 其他可选属性
}

// 定义PE目标
def PE : Target {
  // 指定目标的名称
  let InstructionSet = PEInstrInfo;
}

// 定义寄存器类
class RegisterClass<RegisterType, string Name, list<Register> Regs> {
  let Name = Name;
  let Registers = Regs;
}

// 定义一个简单的寄存器
def R0 : Register;
def R1 : Register;

// 定义寄存器类
def GPR : RegisterClass<"i32", "GPR", [R0, R1]>;

// 定义一个简单的指令
class Instruction {
  let Namespace = "PE";
}

def ADD : Instruction {
  let Inst = "add";
  let Operands = (outs GPR:$dst, GPR:$src1, GPR:$src2);
  let AsmString = "add $dst, $src1, $src2";
}