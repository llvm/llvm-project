// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,MNEMONIC-FIRST
// RUN: llvm-tblgen -gen-asm-matcher -DNO_MNEMONIC_FIRST \
// RUN:   -I %p/../../include %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,NO-MNEMONIC-FIRST

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;

#ifdef NO_MNEMONIC_FIRST
def ArchAsmParser : AsmParser {
  let HasMnemonicFirst = 0;
}
#else
def ArchAsmParser : AsmParser;
#endif

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"Arch", [i32], 32, (add Reg)>;

def FeatureA : SubtargetFeature<"feature-a", "FeatureA", "true",
                                "Enable feature A">;
def HasFeatureA : Predicate<"Subtarget->hasFeatureA()">,
                  AssemblerPredicate<(all_of FeatureA)>;

def AClass : AsmOperandClass {
  let Name = "A";
  let ParserMethod = "parseA";
}

def BClass : AsmOperandClass {
  let Name = "B";
  let ParserMethod = "parseB";
}

def AOperand : Operand<i32> {
  let ParserMatchClass = AClass;
}

def BOperand : Operand<i32> {
  let ParserMatchClass = BClass;
}

class PairInst<string Mnemonic, list<Predicate> RequiredPredicates>
    : Instruction {
  let Size = 4;
  let OutOperandList = (outs);
  let InOperandList = (ins AOperand:$a, BOperand:$b);
  let AsmString = Mnemonic # " $a, $b";
  let Predicates = RequiredPredicates;
}

class SingleInst<string Mnemonic> : Instruction {
  let Size = 4;
  let OutOperandList = (outs);
  let InOperandList = (ins AOperand:$a);
  let AsmString = Mnemonic # " $a";
}

def BarFeature : PairInst<"bar", [HasFeatureA]>;
def BarBase : PairInst<"bar", []>;
def Baz : SingleInst<"baz">;
def FooFeature : PairInst<"foo", [HasFeatureA]>;
def FooBase : PairInst<"foo", []>;

// Bar and foo have identical ordered match sequences. Baz has a distinct
// one-record sequence.
//
// CHECK: static const OperandMatchEntry OperandMatchSequenceTable[5] = {
// CHECK-NEXT: /* Operand Mask, Operand Class, Features */
// CHECK-NEXT: /* 0 */ { {{[0-9]+}} /* {{[0-9]+}} */, MCK_A, AMFBS_HasFeatureA },
// CHECK-NEXT: /* 1 */ { {{[0-9]+}} /* {{[0-9]+}} */, MCK_B, AMFBS_HasFeatureA },
// CHECK-NEXT: /* 2 */ { {{[0-9]+}} /* {{[0-9]+}} */, MCK_A, AMFBS_None },
// CHECK-NEXT: /* 3 */ { {{[0-9]+}} /* {{[0-9]+}} */, MCK_B, AMFBS_None },
// CHECK-NEXT: /* 4 */ { {{[0-9]+}} /* {{[0-9]+}} */, MCK_A, AMFBS_None },
// CHECK: static const OperandMatchMnemonicEntry OperandMatchMnemonicTable[3] = {
// CHECK-NEXT: /* Match Sequence Offset, Mnemonic, Match Sequence Length */
// CHECK-NEXT: { 0, {{[0-9]+}} /* bar */, 4 },
// CHECK-NEXT: { 4, {{[0-9]+}} /* baz */, 1 },
// CHECK-NEXT: { 0, {{[0-9]+}} /* foo */, 4 },

// The generated parser walks each mnemonic's interned sequence in order.
// MNEMONIC-FIRST: auto MnemonicIt =
// MNEMONIC-FIRST: for (unsigned I = MnemonicIt->MatchSequenceOffset;
// MNEMONIC-FIRST-NEXT: I != MatchSequenceEnd; ++I) {
// NO-MNEMONIC-FIRST: auto MnemonicBegin = std::begin(OperandMatchMnemonicTable);
// NO-MNEMONIC-FIRST-NEXT: auto MnemonicEnd = std::end(OperandMatchMnemonicTable);
// NO-MNEMONIC-FIRST-NEXT: if (!Mnemonic.empty()) {
// NO-MNEMONIC-FIRST: for (const OperandMatchMnemonicEntry *MnemonicIt =
// NO-MNEMONIC-FIRST-SAME: MnemonicBegin;
// NO-MNEMONIC-FIRST-NEXT: MnemonicIt != MnemonicEnd; ++MnemonicIt) {
// NO-MNEMONIC-FIRST: for (unsigned I = MnemonicIt->MatchSequenceOffset;
// NO-MNEMONIC-FIRST-NEXT: I != MatchSequenceEnd; ++I) {
// CHECK-NEXT: const OperandMatchEntry &Entry = OperandMatchSequenceTable[I];
