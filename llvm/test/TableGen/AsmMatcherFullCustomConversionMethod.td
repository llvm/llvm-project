// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

class ArchInstruction<string AsmStr, dag OutOperands, dag InOperands> : Instruction {
    let OutOperandList = OutOperands;
    let InOperandList = InOperands;
    let AsmString = AsmStr;
    let AsmMatchConverter = "ConvertInstr";
}

def Instr1 : ArchInstruction<"Instr1", (outs), (ins)>;
def Instr2 : ArchInstruction<"Instr2", (outs), (ins)>;

// Check that the size of ConversionTable is 3

// CHECK:      static const uint8_t ConversionTable[CVT_NUM_SIGNATURES][3] = {
// CHECK-NEXT:   // ConvertCustom_ConvertInstr
// CHECK-NEXT:   { CVT_ConvertInstr, 0, CVT_Done },
// CHECK-NEXT: };

