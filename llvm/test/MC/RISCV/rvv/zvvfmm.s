# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvfmm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvfmm %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvfmm - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vfmmacc.vv v8, v4, v20
# CHECK-INST: vfmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x53]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vfwmmacc.vv v8, v4, v20
# CHECK-INST: vfwmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x57]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vfwmmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfwmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x55]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vfqmmacc.vv v8, v4, v20
# CHECK-INST: vfqmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x5b]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vfqmmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfqmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x59]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vf8wmmacc.vv v8, v4, v20
# CHECK-INST: vf8wmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x5f]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vf8wmmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vf8wmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x5d]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vfwimmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfwimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xe5]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vfqimmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vfqimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xe9]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}

vf8wimmacc.vv v8, v4, v20, v0.scale
# CHECK-INST: vf8wimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xed]
# CHECK-ERROR: instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate){{$}}
