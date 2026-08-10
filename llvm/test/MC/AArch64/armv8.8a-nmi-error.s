// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+nmi   < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=+v8.8a < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -show-encoding               < %s 2>&1 | FileCheck %s --check-prefix=NO_NMI
// RUN: not llvm-mc -triple aarch64 -show-encoding -mattr=-nmi   < %s 2>&1 | FileCheck %s --check-prefix=NO_NMI

msr ALLINT, #1
msr ALLINT, #2
msr ALLINT, x3
mrs x2, ALLINT
mrs x11, icc_nmiar1_el1
msr icc_nmiar1_el1, x12

// CHECK:         error: immediate must be an integer in range [0, 1].
// CHECK-NEXT:    msr ALLINT, #2
// CHECK-NEXT:    ^
// CHECK-NEXT:         error: expected writable system register or pstate
// CHECK-NEXT:    msr icc_nmiar1_el1, x12
// CHECK-NEXT:    ^

// NO_NMI:      error: expected writable system register or pstate
// NO_NMI-NEXT: msr {{allint|ALLINT}}, #1
// NO_NMI-NEXT: ^
// NO_NMI-NEXT: error: expected writable system register or pstate
// NO_NMI-NEXT: msr {{allint|ALLINT}}, #2
// NO_NMI-NEXT: ^
// NO_NMI-NEXT: error: expected writable system register or pstate
// NO_NMI-NEXT: msr {{allint|ALLINT}}, x3
// NO_NMI-NEXT: ^
// NO_NMI-NEXT: error: expected readable system register
// NO_NMI-NEXT: mrs x2, {{allint|ALLINT}}
// NO_NMI-NEXT: ^
// NO_NMI-NEXT: error: expected readable system register
// NO_NMI-NEXT: mrs x11, {{icc_nmiar1_el1|ICC_NMIAR1_EL1}}
// NO_NMI-NEXT: ^
// NO_NMI-NEXT: error: expected writable system register or pstate
// NO_NMI-NEXT: msr {{icc_nmiar1_el1|ICC_NMIAR1_EL1}}, x12
