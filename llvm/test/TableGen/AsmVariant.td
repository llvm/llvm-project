// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s

// Check that specifying AsmVariant works correctly

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo { }

def FooAsmParserVariant : AsmParserVariant {
  let Variant = 0;
  let Name = "Foo";
}

def BarAsmParserVariant : AsmParserVariant {
  let Variant = 1;
  let Name = "Bar";
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParserVariants = [FooAsmParserVariant, BarAsmParserVariant];
}

def Reg : Register<"reg">;

def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

def foo : Instruction {
  let Size = 2;
  let OutOperandList = (outs);
  let InOperandList = (ins);
  let AsmString = "foo";
  let AsmVariantName = "Foo";
}

def BarAlias : InstAlias<"bar", (foo)> {
  string AsmVariantName = "Bar";
}

// CHECK: static const MatchEntry MatchTable0[] = {
// CHECK-NEXT: /* foo */, Arch::foo
// CHECK-NEXT: };

// CHECK: static const MatchEntry MatchTable1[] = {
// CHECK-NEXT: /* bar */, Arch::foo
// CHECK-NEXT: };
