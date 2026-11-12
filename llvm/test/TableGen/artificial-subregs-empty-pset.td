// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s 2>&1 | FileCheck %s

include "llvm/Target/Target.td"

def sub_8 : SubRegIndex<8>;
def sub_16 : SubRegIndex<16>;

class TestReg<string Name, list<Register> SubRegs = []>
    : RegisterWithSubRegs<Name, SubRegs> {
  let Namespace = "Test";
}

def R0B : TestReg<"r0.b"> {
  let isArtificial = 1;
}

def R0W : TestReg<"r0.w", [R0B]> {
  let isArtificial = 1;
  let SubRegIndices = [sub_8];
}

def R0 : TestReg<"r0", [R0W]> {
  let SubRegIndices = [sub_16];
}

def GPR32 : RegisterClass<"Test", [i32], 32, (add R0)>;

def TestTarget : Target;

// CHECK: error: allocatable register class 'GPR32' has no non-artificial register units
