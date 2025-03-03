# RUN: llvm-mc -triple=riscv32 -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-RVI %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-RVI %s
# RUN: llvm-mc -triple=riscv32 -mattr=+c -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-RVIC %s
# RUN: llvm-mc -triple=riscv64 -mattr=+c -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-RVIC %s
# RUN: llvm-mc -triple=riscv32 -mattr=+zca -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-RVIC %s
# RUN: llvm-mc -triple=riscv64 -mattr=+zca -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-RVIC %s
# RUN: llvm-mc -triple=riscv32 -mattr=+e -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-RVE %s
# RUN: llvm-mc -triple=riscv64 -mattr=+e -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-RVE %s
# RUN: llvm-mc -triple=riscv32 -mattr=+ztso -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-TSO %s
# RUN: llvm-mc -triple=riscv64 -mattr=+ztso -filetype=obj < %s | llvm-readobj --file-headers - | FileCheck -check-prefixes=CHECK-TSO %s

# CHECK-RVI:       Flags [ (0x0)
# CHECK-RVI-NEXT:  ]

# CHECK-RVIC:       Flags [ (0x1)
# CHECK-RVIC-NEXT:    EF_RISCV_RVC (0x1)
# CHECK-RVIC-NEXT:  ]

# CHECK-RVE:        Flags [ (0x8)
# CHECK-RVE-NEXT:     EF_RISCV_RVE (0x8)
# CHECK-RVE-NEXT:   ]

# CHECK-TSO:        Flags [ (0x10)
# CHECK-TSO-NEXT:     EF_RISCV_TSO (0x10)
# CHECK-TSO-NEXT:   ]

nop
