# RUN: llvm-mc -triple=riscv32 -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck -check-prefixes=CHECK-RVI %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck -check-prefixes=CHECK-RVI %s
# RUN: llvm-mc -triple=riscv32 -mattr=+c -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck -check-prefixes=CHECK-RVIC %s
# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj < %s | llvm-readobj -file-headers - | FileCheck -check-prefixes=CHECK-RVIC %s

# CHECK-RVI:       Flags [ (0x0)
# CHECK-RVI-NEXT:  ]

# CHECK-RVIC:       Flags [ (0x1)
# CHECK-RVIC-NEXT:    EF_RISCV_RVC (0x1)
# CHECK-RVIC-NEXT:  ]

nop
