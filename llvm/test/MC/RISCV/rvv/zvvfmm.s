# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvfmm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvfmm %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvfmm - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vfmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vfmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x53]

vfwmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vfwmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x57]

vfwmmacc.vv v8, v4, v20, v0.scale # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vfwmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x55]

vfqmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vfqmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x5b]

vfqmmacc.vv v8, v4, v20, v0.scale # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vfqmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x59]

vf8wmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vf8wmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x14,0x42,0x5f]

vf8wmmacc.vv v8, v4, v20, v0.scale # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vf8wmmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x14,0x42,0x5d]

vfwimmacc.vv v8, v4, v20, v0.scale # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vfwimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xe5]

vfqimmacc.vv v8, v4, v20, v0.scale # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vfqimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xe9]

vf8wimmacc.vv v8, v4, v20, v0.scale # expected-error {{instruction requires the following: 'Zvvfmm' (Floating-Point Matrix Multiply-Accumulate)}}
# CHECK-INST: vf8wimmacc.vv v8, v4, v20, v0.scale
# CHECK-ENCODING: [0x57,0x04,0x42,0xed]
