// RUN: llvm-tblgen -gen-register-info -register-info-debug -I %p/../../include %s -o - 2>&1 >/dev/null | FileCheck %s --implicit-check-not=RegisterClass

include "llvm/Target/Target.td"

def lo : SubRegIndex<0, 32>;
def hi : SubRegIndex<32, 32>;

def R0 : Register<"r0">;
def R1 : Register<"r1">;

def A : Register<"a"> {
  let isArtificial = 1;
}

def D0 : Register<"d0"> {
  let SubRegs = [R0, R1];
  let SubRegIndices = [lo, hi];
}

// Same artificial registers can participate in classes of
// registers and their sub-registers.
def RA : RegisterClass<"R", [i32], 32, (add R0, R1, A)>;
def DA : RegisterClass<"D", [i64], 64, (add D0, A)>;

// Make sure there are no DA_and_RA and DA_with_hi kind of
// classes that only contain artificial registers or are inferred
// copies of already existing register classes that only differ
// in the presence of artificial regsiters.
//
// CHECK-LABEL: RegisterClass RA:
// CHECK-NEXT: SpillSize:
// CHECK-NEXT: SpillAlignment:
// CHECK-NEXT: NumRegs:
// CHECK-NEXT: LaneMask:
// CHECK-NEXT: HasDisjunctSubRegs:
// CHECK-NEXT: CoveredBySubRegs:
// CHECK-NEXT: Allocatable:
// CHECK-NEXT: AllocationPriority:
// CHECK-NEXT: BaseClassOrder:
// CHECK-NEXT: Regs: A R0 R1{{$}}
// CHECK-NEXT: SubClasses: RA{{$}}
//
// CHECK-LABEL: RegisterClass DA:
// CHECK-NEXT: SpillSize:
// CHECK-NEXT: SpillAlignment:
// CHECK-NEXT: NumRegs:
// CHECK-NEXT: LaneMask:
// CHECK-NEXT: HasDisjunctSubRegs:
// CHECK-NEXT: CoveredBySubRegs:
// CHECK-NEXT: Allocatable:
// CHECK-NEXT: AllocationPriority:
// CHECK-NEXT: BaseClassOrder:
// CHECK-NEXT: Regs: A D0{{$}}
// CHECK-NEXT: SubClasses: DA{{$}}

def TestTarget : Target;
