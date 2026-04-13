# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xsfvfexpa %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfexpa %s \
# RUN:        | llvm-objdump -d --mattr=+xsfvfexpa - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfexpa %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vfexpa.v v2, v5, v0.t
# CHECK-INST: sf.vfexpa.v v2, v5, v0.t
# CHECK-ENCODING: [0x57,0x11,0x53,0x4c]
# CHECK-ERROR: instruction requires the following: 'Xsfvfexpa' (SiFive Vector Floating-Point Exponential Approximation Instruction){{$}}
# CHECK-UNKNOWN: 4c531157 <unknown>
