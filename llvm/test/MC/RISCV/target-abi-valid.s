# RUN: llvm-mc -triple=riscv32 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s
# RUN: llvm-mc -triple=riscv32 -target-abi ilp32 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s
# RUN: llvm-mc -triple=riscv32 -mattr=+f -target-abi ilp32 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s
# RUN: llvm-mc -triple=riscv32 -mattr=+d -target-abi ilp32 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s
# RUN: llvm-mc -triple=riscv64 -target-abi lp64 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s
# RUN: llvm-mc -triple=riscv64 -mattr=+f -target-abi lp64 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s
# RUN: llvm-mc -triple=riscv64 -mattr=+d -target-abi lp64 -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-NONE %s

# RUN: llvm-mc -triple=riscv32 -mattr=+f -target-abi ilp32f -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-FLOAT-SINGLE %s
# RUN: llvm-mc -triple=riscv32 -mattr=+d -target-abi ilp32f -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-FLOAT-SINGLE %s
# RUN: llvm-mc -triple=riscv64 -mattr=+f -target-abi lp64f -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-FLOAT-SINGLE %s
# RUN: llvm-mc -triple=riscv64 -mattr=+d -target-abi lp64f -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-FLOAT-SINGLE %s

# RUN: llvm-mc -triple=riscv32 -mattr=+d -target-abi ilp32d -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-FLOAT-DOUBLE %s
# RUN: llvm-mc -triple=riscv64 -mattr=+d -target-abi lp64d -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-FLOAT-DOUBLE %s

# RUN: llvm-mc -triple=riscv32 -target-abi ilp32e -filetype=obj < %s \
# RUN:   | llvm-readobj --file-headers - \
# RUN:   | FileCheck -check-prefix=CHECK-RVE %s

# CHECK-NONE:               Flags [ (0x0)
# CHECK-NONE-NEXT:          ]

# CHECK-FLOAT-SINGLE:       Flags [ (0x2)
# CHECK-FLOAT-SINGLE-NEXT:    EF_RISCV_FLOAT_ABI_SINGLE (0x2)
# CHECK-FLOAT-SINGLE-NEXT:  ]

# CHECK-FLOAT-DOUBLE:       Flags [ (0x4)
# CHECK-FLOAT-DOUBLE-NEXT:    EF_RISCV_FLOAT_ABI_DOUBLE (0x4)
# CHECK-FLOAT-DOUBLE-NEXT:  ]

# CHECK-RVE:                Flags [ (0x8)
# CHECK-RVE-NEXT:             EF_RISCV_RVE (0x8)
# CHECK-RVE-NEXT:           ]

nop
