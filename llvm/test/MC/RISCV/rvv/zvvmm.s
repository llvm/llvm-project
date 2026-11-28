# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmm %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmm - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate)}}
# CHECK-INST: vmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xe3]

vwmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate)}}
# CHECK-INST: vwmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xe7]

vqmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate)}}
# CHECK-INST: vqmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xeb]

v8wmmacc.vv v8, v4, v20 # expected-error {{instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate)}}
# CHECK-INST: v8wmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xef]
