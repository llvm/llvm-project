# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvfwmaccqqq %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfwmaccqqq %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvfwmaccqqq - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

sf.vfwmacc.4x4x4 v8, v4, v20 # expected-error {{instruction requires the following: 'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction (4-by-4))}}
# CHECK-INST: sf.vfwmacc.4x4x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x14,0x42,0xf3]

sf.vfwmacc.4x4x4 v0, v4, v20 # expected-error {{instruction requires the following: 'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction (4-by-4))}}
# CHECK-INST: sf.vfwmacc.4x4x4 v0, v4, v20
# CHECK-ENCODING: [0x5b,0x10,0x42,0xf3]
