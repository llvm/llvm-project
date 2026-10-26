// Check that AsmWriter.UseBytecode emits the overflow operands as a program
// table and an interpreter loop, and that the switch form is otherwise kept.

// RUN: llvm-tblgen -gen-asm-writer -I %p/../../include %s \
// RUN:   | FileCheck %s --check-prefix=SWITCH
// RUN: llvm-tblgen -gen-asm-writer -I %p/../../include -DBYTECODE %s \
// RUN:   | FileCheck %s --check-prefix=BYTECODE

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;

def ArchAsmWriter : AsmWriter {
#ifdef BYTECODE
  let UseBytecode = 1;
#endif
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyWriters = [ArchAsmWriter];
}

def R0 : Register<"r0">;
def Reg : RegisterClass<"Reg", [i32], 0, (add R0)>;

def IntOperand : Operand<i32>;

// Instruction I prints I operands, so the commands of every position differ
// and overflow the 64-bit table.
defvar N = 70;

class ArchInst<int Len> : Instruction {
  defvar Names = !foreach(I, !range(Len), "a" # I);
  let OutOperandList = (outs);
  let InOperandList = !dag(ins, !listsplat(IntOperand, Len), Names);
  let AsmString = "i" # Len # " " #
                  !interleave(!foreach(Name, Names, "$" # Name), ", ");
}

foreach Len = !range(1, N) in
  def I#Len : ArchInst<Len>;

// SWITCH-NOT: OpcodeToOffset

// Opcodes without operands to print, such as PHI, keep offset 0.
// BYTECODE:      static const uint16_t OpcodeToOffset[] = {
// BYTECODE-NEXT:   0,	// PHI
// BYTECODE:      static const uint8_t OverflowProgram[] = {
// BYTECODE-NEXT:   /* 0 */ 0,
// BYTECODE-NEXT:   /* 1 */ 1,
// BYTECODE-NEXT:   /* 2 */ 2, 3, 1,
// BYTECODE:      for (uint16_t Idx = OpcodeToOffset[MI->getOpcode()];; ++Idx) {
// BYTECODE-NEXT:   switch (OverflowProgram[Idx]) {
// BYTECODE-NEXT:   default: llvm_unreachable("Unexpected bytecode command.");
// BYTECODE-NEXT:   case 0:
// BYTECODE-NEXT:     llvm_unreachable("Unexpected opcode.");
// BYTECODE-NEXT:     break;
// BYTECODE-NEXT:   case 1:
// BYTECODE-NEXT:     return;
// BYTECODE-NEXT:   case 2:
// BYTECODE-NEXT:     O << ", ";
// BYTECODE-NEXT:     break;
