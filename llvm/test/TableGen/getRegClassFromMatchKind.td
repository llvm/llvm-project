// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s --check-prefixes=CHECK,UNIQUE
// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s -o %t.inc
// RUN: FileCheck %s --check-prefix=REGINFO-ENUM --input-file=%tEnums.inc
// RUN: FileCheck %s --check-prefix=REGINFO-DESC --input-file=%tMCDesc.inc
// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s -DAMBIGUOUS | FileCheck %s --check-prefixes=CHECK

/// This test verifies that TableGen's AsmMatcherEmitter correctly generates
/// the target-independent `getRegClassFromMatchKind` mapping function.
///
/// It specifically verifies that:
/// - Explicit named target RegisterClasses (like `RegClass`) are mapped.
/// - RegisterOperands with custom ParserMatchClass (like `SpecialRegOp`)
///   are mapped to their underlying RegisterClass (unless there are multiple
///   uses of the same ParserMatchClass with different RegisterClasses).
/// - RegClassByHwMode (like `HwModeRC`) are NOT mapped.
/// - Singleton registers (like `Reg` used literally in `bar` instruction)
///   which generate implicit `MCK_Reg` match classes are NOT mapped.
/// - Non-register class operands (like the custom immediate `MyImmOp`
///   in `baz` instruction) are NOT mapped.
/// - All register classes are covered by the generated function (checking enums)


include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def ArchAsmParserVariant : AsmParserVariant {
  let Variant = 0;
  let Name = "Arch";
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParserVariants = [ArchAsmParserVariant];
}

let Namespace = "Arch" in {
  def Reg : Register<"reg">;
  def Reg2 : Register<"reg2">;
}

/// Use "Arch" as namespace to match the Target name.
/// RegisterClasses are expected to be mapped.
def RegClass : RegisterClass<"Arch", [i32], 0, (add Reg, Reg2)>;
def RegClass2 : RegisterClass<"Arch", [i32], 0, (add Reg2)>;
def RegClass3 : RegisterClass<"Arch", [i32], 0, (add Reg)>;

def ModeA : HwMode<[]>;
def ModeB : HwMode<[]>;

/// RegClassByHwMode is NOT expected to be mapped directly.
def HwModeRC : RegClassByHwMode<[ModeA, ModeB], [RegClass, RegClass2]>;

def MyImmAsmOperand : AsmOperandClass {
  let Name = "MyImm";
}

/// Non-register operands are NOT expected to be mapped.
def MyImmOp : Operand<i32> {
  let ParserMatchClass = MyImmAsmOperand;
}

def foo : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins RegClass:$src);
  let AsmString = "foo $src";
  let Namespace = "Arch";
}

/// Singleton registers are NOT expected to be mapped.
def bar : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let AsmString = "bar reg"; // uses Reg literally, making it a singleton register
  let Namespace = "Arch";
}

def baz : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins MyImmOp:$imm);
  let AsmString = "baz $imm"; // uses MyImmAsmOperand
  let Namespace = "Arch";
}

  def SpecialRegAsmOperand : AsmOperandClass {
  let Name = "SpecialRegOperand";
  let ParserMethod = "parseRegister";
  let DiagnosticType = "InvalidSpecialReg";
}

/// RegisterOperands with custom ParserMatchClass are expected to be mapped
/// to their underlying RegisterClass.
def SpecialRegOp : RegisterOperand<RegClass> {
  let ParserMatchClass = SpecialRegAsmOperand;
}

#ifdef AMBIGUOUS
/// This should cause MCK_SpecialRegOperand to be omitted from mapping because it is ambiguous.
def ZAmbiguousRegOp : RegisterOperand<RegClass3> {
  let ParserMatchClass = SpecialRegAsmOperand;
}
#endif

def qux : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins SpecialRegOp:$src);
  let AsmString = "qux $src";
  let Namespace = "Arch";
}

def hwmode_op : RegisterOperand<HwModeRC>;

def qux2 : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins hwmode_op:$src);
  let AsmString = "qux2 $src";
  let Namespace = "Arch";
}

// CHECK:      enum MatchClassKind {
// CHECK-NEXT:   InvalidMatchClass = 0,
// CHECK-NEXT:   OptionalMatchClass = 1,
// CHECK-NEXT:   MCK_LAST_TOKEN = OptionalMatchClass,
// CHECK-NEXT:   MCK_RegClass2, // register class 'RegClass2'
// CHECK-NEXT:   MCK_RegClass3, // register class 'RegClass3,Reg'
// CHECK-NEXT:   MCK_RegClass, // register class 'RegClass'
// CHECK-NEXT:   MCK_LAST_REGISTER = MCK_RegClass,
// CHECK-NEXT:   MCK_RegByHwMode_HwModeRC, // register class by hwmode
// CHECK-NEXT:   MCK_LAST_REGCLASS_BY_HWMODE = MCK_RegByHwMode_HwModeRC,
// CHECK-NEXT:   MCK_Imm, // user defined class 'ImmAsmOperand'
// CHECK-NEXT:   MCK_MyImm, // user defined class 'MyImmAsmOperand'
// CHECK-NEXT:   MCK_SpecialRegOperand, // user defined class 'SpecialRegAsmOperand'
// CHECK-NEXT:   NumMatchClassKinds
// CHECK-NEXT: };

// CHECK:      {{\[\[}}maybe_unused]] static const MCRegisterClass *getRegClassFromMatchKind(MatchClassKind Kind) {
// CHECK-NEXT:   switch (Kind) {
// CHECK-NEXT:   case MCK_RegClass2:
// CHECK-NEXT:     return &ArchMCRegisterClasses[Arch::RegClass2RegClassID];
// CHECK-NEXT:   case MCK_RegClass3:
// CHECK-NEXT:     return &ArchMCRegisterClasses[Arch::RegClass3RegClassID];
// CHECK-NEXT:   case MCK_RegClass:
// CHECK-NEXT:     return &ArchMCRegisterClasses[Arch::RegClassRegClassID];
// UNIQUE-NEXT:  case MCK_SpecialRegOperand:
// UNIQUE-NEXT:    return &ArchMCRegisterClasses[Arch::RegClassRegClassID];
// CHECK-NEXT:   default:
// CHECK-NEXT:     return nullptr;
// CHECK-NEXT:   }
// CHECK-NEXT: }

// REGINFO-ENUM:      // Register classes
// REGINFO-ENUM-EMPTY:
// REGINFO-ENUM-NEXT: namespace Arch {
// REGINFO-ENUM-EMPTY:
// REGINFO-ENUM-NEXT: enum {
// REGINFO-ENUM-NEXT:   RegClassRegClassID = 0,
// REGINFO-ENUM-NEXT:   RegClass2RegClassID = 1,
// REGINFO-ENUM-NEXT:   RegClass3RegClassID = 2,
// REGINFO-ENUM-EMPTY:
// REGINFO-ENUM-NEXT: };

// REGINFO-DESC:      extern const MCRegisterClass ArchMCRegisterClasses[] = {
// REGINFO-DESC-NEXT:   { RegClass, RegClassBits, 20, 2, sizeof(RegClassBits), Arch::RegClassRegClassID, 32, 1, true, false },
// REGINFO-DESC-NEXT:   { RegClass2, RegClass2Bits, 0, 1, sizeof(RegClass2Bits), Arch::RegClass2RegClassID, 32, 1, true, false },
// REGINFO-DESC-NEXT:   { RegClass3, RegClass3Bits, 10, 1, sizeof(RegClass3Bits), Arch::RegClass3RegClassID, 32, 1, true, false },
// REGINFO-DESC-NEXT: };
