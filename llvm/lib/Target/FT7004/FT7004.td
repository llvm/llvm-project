// 包含必要的 LLVM 定义文件
include "llvm/Target/Target.td"
include "FT7004InstrInfo.td"
include "FT7004CallingConv.td"

def : ProcessorModel<"FT7004", NoSchedModel, []>;

def FT7004InstrInfo : InstrInfo;

def FT7004 : Target {
  let InstructionSet = FT7004InstrInfo;
}
