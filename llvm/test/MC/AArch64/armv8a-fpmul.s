// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+fp16fml < %s | FileCheck %s --check-prefix=CHECK
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-fullfp16,+fp16fml < %s | FileCheck %s --check-prefix=CHECK
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOFP16FML
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a,+fullfp16 < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOFP16FML
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a,+fp16fml,-fullfp16 < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOFP16FML
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a,+fp16fml,-neon < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FP16FML-NOR-NEON
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a,-neon < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-FP16FML-NOR-NEON

//------------------------------------------------------------------------------
// ARMV8.2-A Floating Point Multiplication
//------------------------------------------------------------------------------

FMLAL  V0.2S, V1.2H, V2.2H
FMLSL  V0.2S, V1.2H, V2.2H
FMLAL  V0.4S, V1.4H, V2.4H
FMLSL  V0.4S, V1.4H, V2.4H
FMLAL2  V0.2S, V1.2H, V2.2H
FMLSL2  V0.2S, V1.2H, V2.2H
FMLAL2  V0.4S, V1.4H, V2.4H
FMLSL2  V0.4S, V1.4H, V2.4H

//CHECK:  fmlal v0.2s, v1.2h, v2.2h     // encoding: [0x20,0xec,0x22,0x0e]
//CHECK:  fmlsl v0.2s, v1.2h, v2.2h     // encoding: [0x20,0xec,0xa2,0x0e]
//CHECK:  fmlal v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xec,0x22,0x4e]
//CHECK:  fmlsl v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xec,0xa2,0x4e]
//CHECK:  fmlal2  v0.2s, v1.2h, v2.2h     // encoding: [0x20,0xcc,0x22,0x2e]
//CHECK:  fmlsl2  v0.2s, v1.2h, v2.2h     // encoding: [0x20,0xcc,0xa2,0x2e]
//CHECK:  fmlal2  v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xcc,0x22,0x6e]
//CHECK:  fmlsl2  v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xcc,0xa2,0x6e]

//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}

//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}

# Checks with the maximum index value 7:
fmlal  V0.2s, v1.2h, v2.h[7]
fmlsl  V0.2s, v1.2h, v2.h[7]
fmlal  V0.4s, v1.4h, v2.h[7]
fmlsl  V0.4s, v1.4h, v2.h[7]
fmlal2  V0.2s, v1.2h, v2.h[7]
fmlsl2  V0.2s, v1.2h, v2.h[7]
fmlal2  V0.4s, v1.4h, v2.h[7]
fmlsl2  V0.4s, v1.4h, v2.h[7]

# Some more checks with a different index bit pattern to catch
# incorrect permutations of the index (decimal 7 is 0b111):
fmlal  V0.2s, v1.2h, v2.h[5]
fmlsl  V0.2s, v1.2h, v2.h[5]
fmlal  V0.4s, v1.4h, v2.h[5]
fmlsl  V0.4s, v1.4h, v2.h[5]
fmlal2  V0.2s, v1.2h, v2.h[5]
fmlsl2  V0.2s, v1.2h, v2.h[5]
fmlal2  V0.4s, v1.4h, v2.h[5]
fmlsl2  V0.4s, v1.4h, v2.h[5]

//CHECK: fmlal v0.2s, v1.2h, v2.h[7]   // encoding: [0x20,0x08,0xb2,0x0f]
//CHECK: fmlsl v0.2s, v1.2h, v2.h[7]   // encoding: [0x20,0x48,0xb2,0x0f]
//CHECK: fmlal v0.4s, v1.4h, v2.h[7]   // encoding: [0x20,0x08,0xb2,0x4f]
//CHECK: fmlsl v0.4s, v1.4h, v2.h[7]   // encoding: [0x20,0x48,0xb2,0x4f]
//CHECK: fmlal2  v0.2s, v1.2h, v2.h[7]   // encoding: [0x20,0x88,0xb2,0x2f]
//CHECK: fmlsl2  v0.2s, v1.2h, v2.h[7]   // encoding: [0x20,0xc8,0xb2,0x2f]
//CHECK: fmlal2  v0.4s, v1.4h, v2.h[7]   // encoding: [0x20,0x88,0xb2,0x6f]
//CHECK: fmlsl2  v0.4s, v1.4h, v2.h[7]   // encoding: [0x20,0xc8,0xb2,0x6f]

//CHECK:  fmlal v0.2s, v1.2h, v2.h[5]   // encoding: [0x20,0x08,0x92,0x0f]
//CHECK:  fmlsl v0.2s, v1.2h, v2.h[5]   // encoding: [0x20,0x48,0x92,0x0f]
//CHECK:  fmlal v0.4s, v1.4h, v2.h[5]   // encoding: [0x20,0x08,0x92,0x4f]
//CHECK:  fmlsl v0.4s, v1.4h, v2.h[5]   // encoding: [0x20,0x48,0x92,0x4f]
//CHECK:  fmlal2  v0.2s, v1.2h, v2.h[5]   // encoding: [0x20,0x88,0x92,0x2f]
//CHECK:  fmlsl2  v0.2s, v1.2h, v2.h[5]   // encoding: [0x20,0xc8,0x92,0x2f]
//CHECK:  fmlal2  v0.4s, v1.4h, v2.h[5]   // encoding: [0x20,0x88,0x92,0x6f]
//CHECK:  fmlsl2  v0.4s, v1.4h, v2.h[5]   // encoding: [0x20,0xc8,0x92,0x6f]

//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}
//CHECK-NOFP16FML: error: instruction requires: fp16fml{{$}}

//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}
//CHECK-NO-FP16FML-NOR-NEON: error: instruction requires: fp16fml neon{{$}}

