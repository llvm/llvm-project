// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }
def ArchAsmParser : AsmParser {
  let EmitTokenDiagnosticTypes = true;
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

class ArchInstruction<string AsmStr, dag OutOperands, dag InOperands> : Instruction {
    let OutOperandList = OutOperands;
    let InOperandList = InOperands;
    let AsmString = AsmStr;
}

def Instr1 : ArchInstruction<"Instr1 [$foo]", (outs RegClass:$foo), (ins)> {
  bits<1> foo;
}
def Instr2 : ArchInstruction<"Instr2 shift $foo", (outs RegClass:$foo), (ins)> {
  bits<1> foo;
}
// Ensure there are match class kinds for the tokens `[`, `]`, and `shift`
// CHECK-LABEL: enum MatchClassKind
// CHECK:       OptionalMatchClass = 1,
// CHECK-NEXT:  MCK__91_, // '['
// CHECK-NEXT:  MCK__93_, // ']'
// CHECK-NEXT:  MCK_shift, // 'shift'
// CHECK-NEXT:  MCK_LAST_TOKEN = MCK_shift,

// Ensure there are diagnostic strings for Match_InvalidToken* failures
// CHECK-LABEL: getMatchKindDiag
// CHECK:      (ArchAsmParser::ArchMatchResultTy MatchResult) {
// CHECK-NEXT:   switch (MatchResult) {
// CHECK-NEXT:   case ArchAsmParser::Match_InvalidToken_91_:
// CHECK-NEXT:     return "expected '['";
// CHECK-NEXT:   case ArchAsmParser::Match_InvalidToken_93_:
// CHECK-NEXT:     return "expected ']'";
// CHECK-NEXT:   case ArchAsmParser::Match_InvalidTokenshift:
// CHECK-NEXT:     return "expected 'shift'";

// Ensure there are Match_InvalidToken* instances for each token kind
// CHECK-LABEL: getDiagKindFromTokenClass
// CHECK:      (MatchClassKind Kind) {
// CHECK-NEXT:   switch (Kind) {
// CHECK-NEXT:   default:
// CHECK-NEXT:     return MCTargetAsmParser::Match_InvalidOperand;
// CHECK-NEXT:   case MCK__91_:
// CHECK-NEXT:     return ArchAsmParser::Match_InvalidToken_91_;
// CHECK-NEXT:   case MCK__93_:
// CHECK-NEXT:     return ArchAsmParser::Match_InvalidToken_93_;
// CHECK-NEXT:   case MCK_shift:
// CHECK-NEXT:     return ArchAsmParser::Match_InvalidTokenshift;
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Ensure that we call `getDiagKindFromTokenClass` when a token is expected but not found
// CHECK-LABEL: validateOperandClass
// CHECK:      (MCParsedAsmOperand &GOp, MatchClassKind Kind, const MCSubtargetInfo &STI) {
// CHECK-NEXT:   ArchOperand &Operand = (ArchOperand &)GOp;
// CHECK-NEXT:   if (Kind == InvalidMatchClass)
// CHECK-NEXT:     return MCTargetAsmParser::Match_InvalidOperand;
// CHECK-EMPTY:
// CHECK-NEXT:   if (Kind <= MCK_LAST_TOKEN) {
// CHECK-NEXT:     if (Operand.isToken() &&
// CHECK-NEXT:         isSubclass(matchTokenString(Operand.getToken()), Kind))
// CHECK-NEXT:       return MCTargetAsmParser::Match_Success;
// CHECK-NEXT:     return getDiagKindFromTokenClass(Kind);
// CHECK-NEXT:   }
