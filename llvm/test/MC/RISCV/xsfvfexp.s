# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xsfvfexp32e %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfexp32e %s \
# RUN:        | llvm-objdump -d --mattr=+xsfvfexp32e - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfexp32e %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+xsfvfexp16e %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfexp16e %s \
# RUN:        | llvm-objdump -d --mattr=+xsfvfexp16e - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+xsfvfexp16e %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN
# RUN: llvm-mc -triple=riscv64 -show-encoding --mattr=+zvfbfmin,+xsfvfbfexp16e %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zvfbfmin,+xsfvfbfexp16e %s \
# RUN:        | llvm-objdump -d --mattr=+xsfvfbfexp16e - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv64 -filetype=obj --mattr=+zvfbfmin,+xsfvfbfexp16e %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

sf.vfexp.v v2, v5, v0.t
# CHECK-INST: sf.vfexp.v v2, v5, v0.t
# CHECK-ENCODING: [0x57,0x91,0x53,0x4c]
# CHECK-ERROR: instruction requires the following: 'Xsfvfbfexp16e', 'Xsfvfexp16e', or 'Xsfvfexp32e' (SiFive Vector Floating-Point Exponential Function Instruction){{$}}
# CHECK-UNKNOWN: 4c539157 <unknown>
