// RUN: not llvm-mc -triple arm -mattr=+fp16fml,+neon -show-encoding < %s 2>&1 | FileCheck %s  --check-prefix=CHECK-ERROR

VFMAL.F16 D0, S1, S2[2]
vfmsl.f16 d0, s1, s2[2]
vfmsl.f16 d0, s1, s2[-1]
vfmal.f16 q0, d1, d2[4]
VFMSL.F16 Q0, D1, D2[4]
vfmal.f16 q0, d1, d2[-1]

//CHECK-ERROR:      error: invalid operand for instruction
//CHECK-ERROR-NEXT: VFMAL.F16 D0, S1, S2[2]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: invalid operand for instruction
//CHECK-ERROR-NEXT: vfmsl.f16 d0, s1, s2[2]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: invalid operand for instruction
//CHECK-ERROR-NEXT: vfmsl.f16 d0, s1, s2[-1]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: invalid operand for instruction
//CHECK-ERROR-NEXT: vfmal.f16 q0, d1, d2[4]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: invalid operand for instruction
//CHECK-ERROR-NEXT: VFMSL.F16 Q0, D1, D2[4]
//CHECK-ERROR-NEXT:                     ^
//CHECK-ERROR-NEXT: error: invalid operand for instruction
//CHECK-ERROR-NEXT: vfmal.f16 q0, d1, d2[-1]
//CHECK-ERROR-NEXT:                     ^
