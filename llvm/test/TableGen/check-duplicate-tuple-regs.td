// RUN: not llvm-tblgen -gen-register-info -I %p/../../include -I %p/Common %s 2>&1 | FileCheck %s

include "reg-with-subregs-common.td"

// CHECK: error: Register tuple redefines register 'R0_R1_R2_R3'
def GPR128_Aligned : RegisterTuples<[sub0, sub1, sub2, sub3],
                                    [(decimate (shl GPR32, 0), 4),
                                     (decimate (shl GPR32, 1), 4),
                                     (decimate (shl GPR32, 2), 4),
                                     (decimate (shl GPR32, 3), 4)]>;
