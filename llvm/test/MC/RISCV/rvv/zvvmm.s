# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+experimental-zvvmm %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+experimental-zvvmm %s \
# RUN:        | llvm-objdump -d --mattr=+experimental-zvvmm - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vmmacc.vv v8, v4, v20
# CHECK-INST: vmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xe3]
# CHECK-ERROR: instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate){{$}}

vwmmacc.vv v8, v4, v20
# CHECK-INST: vwmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xe7]
# CHECK-ERROR: instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate){{$}}

vqmmacc.vv v8, v4, v20
# CHECK-INST: vqmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xeb]
# CHECK-ERROR: instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate){{$}}

v8wmmacc.vv v8, v4, v20
# CHECK-INST: v8wmmacc.vv v8, v4, v20
# CHECK-ENCODING: [0x57,0x04,0x42,0xef]
# CHECK-ERROR: instruction requires the following: 'Zvvmm' (Integer Matrix Multiply-Accumulate){{$}}
