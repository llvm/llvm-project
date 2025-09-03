// HelloWorld.td
include "llvm/Target/Target.td"

// 1) 先定義一個 InstrInfo，供 Target 的 InstructionSet 使用
def MyInstrInfo : InstrInfo;

// 2) Target 必須指定 InstructionSet
def MyTarget : Target {
  let InstructionSet = MyInstrInfo;
}

// ★ 最小需求：至少一個暫存器與一個 RegisterClass
def R0  : Register<"r0">;
def GPR : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

// 指令基底：固定長度與彙編字串欄位
class MyInst<string asmstr> : Instruction {
  let Namespace = "MyTarget";   // ★ 關鍵：歸屬到 MyTarget
  let Size = 4;
  let AsmString = asmstr;
}

// 單一指令，無輸入輸出操作元
def NOP : MyInst<"nop"> {
  let OutOperandList = (outs);
  let InOperandList  = (ins);
}
