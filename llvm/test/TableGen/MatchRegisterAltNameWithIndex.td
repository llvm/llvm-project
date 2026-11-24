// RUN: llvm-tblgen -gen-asm-matcher -I %p/../../include %s | FileCheck %s
// RUN: llvm-tblgen -gen-asm-matcher -DDUPLICATES -I %p/../../include %s | FileCheck %s

include "llvm/Target/Target.td"

def ArchInstrInfo : InstrInfo;
def ArchAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = false;
  let ShouldEmitMatchRegisterAltNameWithIndex = true;
#ifdef DUPLICATES
  let AllowDuplicateRegisterNames = true;
#endif
}
def Arch : Target {
  let InstructionSet = ArchInstrInfo;
  let AssemblyParsers = [ArchAsmParser];
}

let Namespace = "Arch" in {
def ABI0 : RegAltNameIndex;
let FallbackRegAltNameIndex = ABI0 in
def ABI1 : RegAltNameIndex;

class ArchReg<string name, list<string> names,
              list<RegAltNameIndex> indices = [ABI0, ABI1]> : Register<name> {
  let AltNames = names;
  let RegAltNameIndices = indices;
}

// Shared spellings select different registers by index.
def R0 : ArchReg<"r0", ["arg", "tmp"]>;
def R1 : ArchReg<"r1", ["tmp", "arg"]>;
// Ignore empty and unindexed names.
def R2 : ArchReg<"r2", ["", "extra", "unindexed"]>;
// Allow multiple names per index.
def R3 : ArchReg<"r3", ["alias0", "alias1"], [ABI0, ABI0]>;
#ifdef DUPLICATES
def R0_64 : ArchReg<"r0", ["arg", "tmp"]>;
#endif
}

def GPR : RegisterClass<"Arch", [i32], 32, (add R0, R1, R2, R3)>;

// CHECK-LABEL: static MCRegister MatchRegisterAltName(StringRef Name, unsigned AltIdx) {
// CHECK-NEXT:    switch (AltIdx) {
// CHECK-NEXT:    default: break;
// CHECK:         case Arch::ABI0: {
// CHECK-DAG:       return Arch::R0;  // "arg"
// CHECK-DAG:       return Arch::R1;  // "tmp"
// CHECK-DAG:       return Arch::R3;  // "alias0"
// CHECK-DAG:       return Arch::R3;  // "alias1"
// CHECK-NOT:     return Arch::R2;
// CHECK:         case Arch::ABI1: {
// CHECK-DAG:       return Arch::R0;  // "tmp"
// CHECK-DAG:       return Arch::R1;  // "arg"
// CHECK-DAG:       return Arch::R2;  // "extra"
// CHECK-NOT:     return Arch::R3;
// CHECK-NOT:     MatchRegisterAltName(Name,
// CHECK:         return Arch::NoRegister;
// CHECK-NEXT:  }
