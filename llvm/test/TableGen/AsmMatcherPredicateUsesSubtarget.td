// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;
def Arch : Target {
  let InstructionSet = ArchInstrInfo;
}

let Namespace = "Arch" in {
  def R0 : Register<"r0">;
}

def GPR : RegisterClass<"Arch", [i32], 32, (add R0)>;

def PlainImmAsmOp : AsmOperandClass {
  let Name = "PlainImm";
}

def SubtargetImmAsmOp : AsmOperandClass {
  let Name = "SubtargetImm";
  let PredicateMethod = "isSubtargetValidImm";
  let PredicateUsesSubtarget = true;
}

def DefaultSubtargetAsmOp : AsmOperandClass {
  let Name = "DefaultSubtargetOp";
  let PredicateUsesSubtarget = true;
}

def PlainImm : Operand<i32> {
  let ParserMatchClass = PlainImmAsmOp;
}

def SubtargetImm : Operand<i32> {
  let ParserMatchClass = SubtargetImmAsmOp;
}

def DefaultSubtargetOp : Operand<i32> {
  let ParserMatchClass = DefaultSubtargetAsmOp;
}

def InstA : Instruction {
  let OutOperandList = (outs GPR:$rd);
  let InOperandList = (ins PlainImm:$a, SubtargetImm:$b, DefaultSubtargetOp:$c);
  let AsmString = "foo $rd, $a, $b, $c";
}

// CHECK-LABEL: static unsigned validateOperandClass(MCParsedAsmOperand &GOp, MatchClassKind Kind, const MCSubtargetInfo &STI) {
// CHECK:       case MCK_DefaultSubtargetOp: {
// CHECK-NEXT:    DiagnosticPredicate DP(Operand.isDefaultSubtargetOp(STI));
// CHECK:       case MCK_PlainImm: {
// CHECK-NEXT:    DiagnosticPredicate DP(Operand.isPlainImm());
// CHECK:       case MCK_SubtargetImm: {
// CHECK-NEXT:    DiagnosticPredicate DP(Operand.isSubtargetValidImm(STI));
