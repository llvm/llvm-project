# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+v,+xsfvfwmaccqqq %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfwmaccqqq %s \
# RUN:        | llvm-objdump -d --mattr=+v,+xsfvfwmaccqqq - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+v,+xsfvfwmaccqqq %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vfwmacc.4x4x4 v8, v4, v20
# CHECK-INST: sf.vfwmacc.4x4x4 v8, v4, v20
# CHECK-ENCODING: [0x5b,0x14,0x42,0xf3]
# CHECK-ERROR: instruction requires the following: 'XSfvfwmaccqqq' (SiFive Matrix Multiply Accumulate Instruction and 4-by-4))
# CHECK-UNKNOWN: f342145b <unknown>
