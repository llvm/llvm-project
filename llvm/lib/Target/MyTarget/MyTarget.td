// HelloWorld.td
include "llvm/Target/Target.td"

// 定義一個虛擬暫存器
def R0 : Register<"r0">;

// 定義暫存器類別（必需的）
def MyRegClass : RegisterClass<"MyTarget", [i32], 32, (add R0)>;

// 定義指令資訊（必需的）
def MyTargetInstrInfo : InstrInfo;

// 定義目標（現在包含必要的組件）
def MyTarget : Target {
  let InstructionSet = MyTargetInstrInfo;
}

// 指令基底：固定長度與彙編字串欄位
class MyInst<string asmstr> : Instruction {
  let Size = 4;
  let AsmString = asmstr;
  let Namespace = "MyTarget";
  bits<32> Inst;            // ★ 必須：宣告用來存放編碼的欄位
}

// 單一指令，無輸入輸出操作元
def NOP : MyInst<"nop"> {
  let Inst = 0b00000000000000000000000000000000; // 32-bit 全 0
  let OutOperandList = (outs);
  let InOperandList  = (ins);
}
