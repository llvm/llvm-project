// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o - 2>&1 >/dev/null | FileCheck %s
include "llvm/Target/Target.td"
def TestTarget : Target;

def lo : SubRegIndex<32, 0>;
def hi : SubRegIndex<32, 32>;

// One superreg with two subregs.
def Alo : Register<"">;
def Ahi : Register<"">;
def A : Register<""> {
  let SubRegs = [Alo, Ahi];
  let SubRegIndices = [lo, hi];
}

// Same but the subregs alias.
def Blo : Register<"">;
def Bhi : Register<""> {
  let Aliases = [Blo];
}
def B : Register<""> {
  let SubRegs = [Blo, Bhi];
  let SubRegIndices = [lo, hi];
}

// Same but the superreg has an alias.
def Clo : Register<"">;
def Chi : Register<"">;
def D : Register<"">;
def C : Register<""> {
  let SubRegs = [Clo, Chi];
  let SubRegIndices = [lo, hi];
  let Aliases = [D];
}

def TestRC : RegisterClass<"Test", [i64], 0, (add A, B)>;

// CHECK-LABEL: Register A:
// CHECK: SubReg hi = Ahi
// CHECK: SubReg lo = Alo
// CHECK: RegUnit 0
// CHECK: RegUnit 1

// CHECK-LABEL: Register Ahi:
// CHECK: RegUnit 1

// CHECK-LABEL: Register Alo:
// CHECK: RegUnit 0

// CHECK-LABEL: Register B:
// CHECK: SubReg hi = Bhi
// CHECK: SubReg lo = Blo
// CHECK: RegUnit 2
// CHECK: RegUnit 3
// CHECK: RegUnit 4

// CHECK-LABEL: Register Bhi:
// CHECK: RegUnit 3
// CHECK: RegUnit 4

// CHECK-LABEL: Register Blo:
// CHECK: RegUnit 2
// CHECK: RegUnit 3

// CHECK-LABEL: Register C:
// CHECK: SubReg hi = Chi
// CHECK: SubReg lo = Clo
// CHECK: RegUnit 5
// CHECK: RegUnit 6
// CHECK: RegUnit 7

// CHECK-LABEL: Register Chi:
// CHECK: RegUnit 6

// CHECK-LABEL: Register Clo:
// CHECK: RegUnit 5

// CHECK-LABEL: Register D:
// CHECK: RegUnit 7
// CHECK: RegUnit 8
