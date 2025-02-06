// RUN: llvm-mc -triple aarch64 -mattr=+lse -mattr=+lsui -show-encoding %s  | FileCheck %s
// RUN: not llvm-mc -triple aarch64 -mattr=+lsui -show-encoding %s 2>&1  | FileCheck %s --check-prefix=ERROR

_func:
// CHECK: _func:

//------------------------------------------------------------------------------
// CAS(P)T instructions
//------------------------------------------------------------------------------
cas w26, w28, [x21]
// CHECK: cas  w26, w28, [x21]
// ERROR: error: instruction requires: lse

casl w26, w28, [x21]
// CHECK: casl  w26, w28, [x21]
// ERROR: error: instruction requires: lse

casa w26, w28, [x21]
// CHECK: casa  w26, w28, [x21]
// ERROR: error: instruction requires: lse

casal w26, w28, [x21]
// CHECK: casal  w26, w28, [x21]
// ERROR: error: instruction requires: lse

casp w26, w27, w28, w29, [x21]
// CHECK: casp w26, w27, w28, w29, [x21]
// ERROR: error: instruction requires: lse

caspl w26, w27, w28, w29, [x21]
// CHECK: caspl w26, w27, w28, w29, [x21]
// ERROR: error: instruction requires: lse

caspa w26, w27, w28, w29, [x21]
// CHECK: caspa w26, w27, w28, w29, [x21]
// ERROR: error: instruction requires: lse

caspal w26, w27, w28, w29, [x21]
// CHECK: caspal w26, w27, w28, w29, [x21]
// ERROR: error: instruction requires: lse
