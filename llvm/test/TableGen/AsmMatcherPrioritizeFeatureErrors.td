// RUN: not llvm-tblgen -gen-asm-matcher -I %p/../../include %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not="error:"

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;

def ArchAsmParser : AsmParser {
  let ReportMultipleNearMisses = false;
  let PrioritizeFeatureInMultiMismatchFallback = true;
}

def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

def Reg : Register<"reg">;
def RegClass : RegisterClass<"foo", [i32], 0, (add Reg)>;

// CHECK: error: 'PrioritizeFeatureInMultiMismatchFallback' requires 'ReportMultipleNearMisses' to be set