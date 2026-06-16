// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s

// Check that specifying AsmVariant works correctly

include "llvm/Target/Target.td"

def ArchParser : AsmParser {
  let HasMnemonicFirst = 0;
}

def Arch : Target {
  let AssemblyParsers = [ArchParser];
}

def Reg : Register<"reg">;

def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

def OptinalAsmOperand :AsmOperandClass {
  let Name = "OptinalAsmOperand";
  let RenderMethod = "addOptinalAsmOperand";
  let IsOptional = 1;
}

def OptinalOperand : Operand<i32> {
  let OperandNamespace = "Arch";
  let OperandType = "OPERAND_INPUT_MODS";
  let PrintMethod = "printOptinalOperand";
  let ParserMatchClass = OptinalAsmOperand;
}

class InstCommonBase {
  field bits<64> Inst;
}

def foo : Instruction, InstCommonBase {
  bits<32> optinalOp;

  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins OptinalOperand:$optinalOp);
  let AsmString = "$optinalOp\tfoo";
  let Namespace = "Arch";

  let Inst{22 - 10} = optinalOp{12 - 0};
}

// CHECK: if (OptionalOperandsMask[*(p + 1)]) {

// CHECK: SmallBitVector OptionalOperandsMask(2);

// CHECK: OptionalOperandsMask.reset(0, 2);

// CHECK: if (Formal == InvalidMatchClass) {
// CHECK-NEXT:   OptionalOperandsMask.set(FormalIdx, 2);
// CHECK: if (isSubclass(Formal, OptionalMatchClass)) {
// CHECK-NEXT:   OptionalOperandsMask.set(FormalIdx);
// CHECK: if (Diag == Match_InvalidOperand && isSubclass(Formal, OptionalMatchClass)) {
// CHECK-NEXT:   OptionalOperandsMask.set(FormalIdx);

// CHECK: for (unsigned i = 0, NumDefaults = 0; i < 2; ++i) {
// CHECK-NEXT:   NumDefaults += (OptionalOperandsMask[i] ? 1 : 0);
// CHECK-NEXT:   DefaultsOffset[i + 1] = NumDefaults;
// CHECK-NEXT: }
