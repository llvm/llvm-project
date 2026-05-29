// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s --check-prefix=ASMMATCHER
// RUN: llvm-tblgen -gen-register-info -I %p/../../include %s -o %t.inc
// RUN: FileCheck %s --check-prefix=REGINFO-ENUM --input-file=%tEnums.inc
// RUN: FileCheck %s --check-prefix=REGINFO-DESC --input-file=%tMCDesc.inc

/// This test verifies that TableGen's AsmMatcherEmitter correctly generates
/// the target-independent `getRegClassFromMatchKind` mapping function.
///
/// It specifically verifies that:
/// - Explicit named target RegisterClasses (like `RegClass`) are mapped.
/// - Singleton registers (like `Reg` used literally in `bar` instruction)
///    which generate implicit `MCK_Reg` match classes are NOT mapped.
/// - Non-register class operands (like the custom immediate `MyImmOp`
///    in `baz` instruction) are NOT mapped.
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
def RegClass : RegisterClass<"Arch", [i32], 0, (add Reg, Reg2)>;

def MyImmAsmOperand : AsmOperandClass {
  let Name = "MyImm";
}

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

// ASMMATCHER:      enum MatchClassKind {
// ASMMATCHER-NEXT:   InvalidMatchClass = 0,
// ASMMATCHER-NEXT:   OptionalMatchClass = 1,
// ASMMATCHER-NEXT:   MCK_LAST_TOKEN = OptionalMatchClass,
// ASMMATCHER-NEXT:   MCK_Reg, // register class 'Reg'
// ASMMATCHER-NEXT:   MCK_RegClass, // register class 'RegClass'
// ASMMATCHER-NEXT:   MCK_LAST_REGISTER = MCK_RegClass,
// ASMMATCHER-NEXT:   MCK_Imm, // user defined class 'ImmAsmOperand'
// ASMMATCHER-NEXT:   MCK_MyImm, // user defined class 'MyImmAsmOperand'
// ASMMATCHER-NEXT:   NumMatchClassKinds
// ASMMATCHER-NEXT: };

// ASMMATCHER:      {{\[\[}}maybe_unused]] static const MCRegisterClass *getRegClassFromMatchKind(MatchClassKind Kind) {
// ASMMATCHER-NEXT:   switch (Kind) {
// ASMMATCHER-NEXT:   case MCK_RegClass:
// ASMMATCHER-NEXT:     return &ArchMCRegisterClasses[Arch::RegClassRegClassID];
// ASMMATCHER-NEXT:   default:
// ASMMATCHER-NEXT:     return nullptr;
// ASMMATCHER-NEXT:   }
// ASMMATCHER-NEXT: }

// REGINFO-ENUM:      // Register classes
// REGINFO-ENUM-EMPTY:
// REGINFO-ENUM-NEXT: namespace Arch {
// REGINFO-ENUM-EMPTY:
// REGINFO-ENUM-NEXT: enum {
// REGINFO-ENUM-NEXT:   RegClassRegClassID = 0,
// REGINFO-ENUM-EMPTY:
// REGINFO-ENUM-NEXT: };

// REGINFO-DESC:      extern const MCRegisterClass ArchMCRegisterClasses[] = {
// REGINFO-DESC-NEXT:   { RegClass, RegClassBits, 0, 2, sizeof(RegClassBits), Arch::RegClassRegClassID, 32, 1, true, false },
// REGINFO-DESC-NEXT: };
