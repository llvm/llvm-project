// RUN: llvm-mc -triple arm -mattr=+fp16fml,+neon -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mattr=+fp16fml,+neon -show-encoding < %s | FileCheck %s --check-prefix=CHECK-T32
// RUN: llvm-mc -triple arm -mattr=-fullfp16,+fp16fml,+neon -show-encoding < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple thumb -mattr=-fullfp16,+fp16fml,+neon -show-encoding < %s | FileCheck %s --check-prefix=CHECK-T32

// RUN: not llvm-mc -triple arm -mattr=+v8.2a -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML-NOR-NEON < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.2a -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML-NOR-NEON < %t %s

// RUN: not llvm-mc -triple arm -mattr=+v8.2a,+neon -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.2a,+neon -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s

// RUN: not llvm-mc -triple arm -mattr=+v8.2a,+neon,+fp16fml,-fp16fml -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.2a,+neon,+fp16fml,-fp16fml -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s

// RUN: not llvm-mc -triple arm -mattr=+v8.2a,+neon,+fullfp16 -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.2a,+neon,+fullfp16 -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s

// RUN: not llvm-mc -triple arm -mattr=+v8.2a,+neon,+fp16fml,-fullfp16 -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.2a,+neon,+fp16fml,-fullfp16 -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-FP16FML < %t %s

// RUN: not llvm-mc -triple arm -mattr=+v8.2a,+fp16fml -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-NEON < %t %s
// RUN: not llvm-mc -triple thumb -mattr=+v8.2a,+fp16fml -show-encoding < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-NEON < %t %s

VFMAL.F16 D0, S1, S2
vfmsl.f16 d0, s1, s2
vfmal.f16 q0, d1, d2
VFMSL.F16 Q0, D1, D2

VFMAL.F16 D0, S1, S2[1]
vfmsl.f16 d0, s1, s2[1]
vfmal.f16 q0, d1, d2[3]
VFMSL.F16 Q0, D1, D2[3]

//CHECK: vfmal.f16 d0, s1, s2      @ encoding: [0x91,0x08,0x20,0xfc]
//CHECK: vfmsl.f16 d0, s1, s2      @ encoding: [0x91,0x08,0xa0,0xfc]
//CHECK: vfmal.f16 q0, d1, d2      @ encoding: [0x52,0x08,0x21,0xfc]
//CHECK: vfmsl.f16 q0, d1, d2      @ encoding: [0x52,0x08,0xa1,0xfc]

//CHECK:  vfmal.f16 d0, s1, s2[1]   @ encoding: [0x99,0x08,0x00,0xfe]
//CHECK:  vfmsl.f16 d0, s1, s2[1]   @ encoding: [0x99,0x08,0x10,0xfe]
//CHECK:  vfmal.f16 q0, d1, d2[3]   @ encoding: [0x7a,0x08,0x01,0xfe]
//CHECK:  vfmsl.f16 q0, d1, d2[3]   @ encoding: [0x7a,0x08,0x11,0xfe]

//CHECK-T32:  vfmal.f16 d0, s1, s2      @ encoding: [0x20,0xfc,0x91,0x08]
//CHECK-T32:  vfmsl.f16 d0, s1, s2      @ encoding: [0xa0,0xfc,0x91,0x08]
//CHECK-T32:  vfmal.f16 q0, d1, d2      @ encoding: [0x21,0xfc,0x52,0x08]
//CHECK-T32:  vfmsl.f16 q0, d1, d2      @ encoding: [0xa1,0xfc,0x52,0x08]

//CHECK-T32:  vfmal.f16 d0, s1, s2[1]   @ encoding: [0x00,0xfe,0x99,0x08]
//CHECK-T32:  vfmsl.f16 d0, s1, s2[1]   @ encoding: [0x10,0xfe,0x99,0x08]
//CHECK-T32:  vfmal.f16 q0, d1, d2[3]   @ encoding: [0x01,0xfe,0x7a,0x08]
//CHECK-T32:  vfmsl.f16 q0, d1, d2[3]   @ encoding: [0x11,0xfe,0x7a,0x08]

//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}
//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}
//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}
//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}
//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}
//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}
//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}
//CHECK-NO-FP16FML: instruction requires: full half-float fml{{$}}

//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}
//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}
//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}
//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}
//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}
//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}
//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}
//CHECK-NO-FP16FML-NOR-NEON: instruction requires: full half-float fml NEON{{$}}

//CHECK-NO-NEON: instruction requires: NEON{{$}}
//CHECK-NO-NEON: instruction requires: NEON{{$}}
//CHECK-NO-NEON: instruction requires: NEON{{$}}
//CHECK-NO-NEON: instruction requires: NEON{{$}}
//CHECK-NO-NEON: instruction requires: NEON{{$}}
//CHECK-NO-NEON: instruction requires: NEON{{$}}
//CHECK-NO-NEON: instruction requires: NEON{{$}}
//CHECK-NO-NEON: instruction requires: NEON{{$}}

