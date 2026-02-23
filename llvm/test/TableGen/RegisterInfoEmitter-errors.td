// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s -DERROR0 2>&1 | FileCheck %s --check-prefix=ERROR0
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s -DERROR1 2>&1 | FileCheck %s --check-prefix=ERROR1
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s -DERROR2 2>&1 | FileCheck %s --check-prefix=ERROR2
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s -DERROR3 2>&1 | FileCheck %s --check-prefix=ERROR3
// RUN: not llvm-tblgen -gen-register-info -I %p/../../include %s -DERROR4 2>&1 | FileCheck %s --check-prefix=ERROR4
// RUN: not llvm-tblgen -gen-register-bank -I %p/../../include %s -DERROR5 2>&1 | FileCheck %s --check-prefix=ERROR5

include "llvm/Target/Target.td"

def MyTarget : Target;

foreach Index = 0-15 in {
  def R#Index : Register <"r"#Index>;
}

def GPR32 : RegisterClass<"MyTarget", [i32], 32, (sequence "R%u", 0, 15)>;

def sub0 : SubRegIndex<32, 0>;
def sub1 : SubRegIndex<32, 32>;

#ifdef ERROR0
// ERROR0: error: assertion failed: RegTypes list must not be empty
// ERROR0: [[@LINE+1]]:5: error: assertion failed in this record
def ClassA : RegisterClass<"MyTarget", [], 32, (add R0)>;
#endif

#ifdef ERROR1
// ERROR1: error: assertion failed: SubRegs and SubRegIndices must have the same size
// ERROR1: error: assertion failed: Tuples must have at least 2 sub-registers
// ERROR1: [[@LINE+1]]:5: error: assertion failed in this record
def Tuple0 : RegisterTuples<[sub0, sub1], [(shl GPR32, 0)]>;
#endif

#ifdef ERROR2
// ERROR2: error: assertion failed: SubRegs and SubRegIndices must have the same size
// ERROR2: [[@LINE+1]]:5: error: assertion failed in this record
def Tuple1 : RegisterTuples<[sub0], [(shl GPR32, 0), (shl GPR32, 1)]>;
#endif

#ifdef ERROR3
// ERROR3: error: assertion failed: SubRegs and SubRegIndices must have the same size
// ERROR3: [[@LINE+1]]:5: error: assertion failed in this record
def R0_R1 : RegisterWithSubRegs<"r0_r1", [R0, R1]>;
#endif

#ifdef ERROR4
// ERROR4: error: assertion failed: SubRegs and SubRegIndices must have the same size
// ERROR4: [[@LINE+1]]:5: error: assertion failed in this record
def R0_R1 : RegisterWithSubRegs<"r0_r1", [R0]> {
    let SubRegIndices = [sub0, sub1];
}
#endif

#ifdef ERROR5
// ERROR5: error: No register banks defined
#endif

