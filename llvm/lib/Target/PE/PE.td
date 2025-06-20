// 包含必要的 LLVM 定义文件
include "llvm/Target/Target.td"//llvm框架层内置的文件
include "PEInstrInfo.td"
include "PECallingConv.td"

def : ProcessorModel<"PE", NoSchedModel, []>; // 定义一个处理器模型

// 定义 PE 指令集信息
def PEInstrInfo : InstrInfo;
// 定义PE目标
def PE : Target {
  // 指定目标的名称
  let InstructionSet = PEInstrInfo;
}
