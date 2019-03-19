; RUN: llc -mtriple=riscv32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -target-abi ilp32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -mattr=+f -target-abi ilp32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -mattr=+d -target-abi ilp32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -target-abi lp64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+f -target-abi lp64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s

define void @nothing() nounwind {
; CHECK-IMP-LABEL: nothing:
; CHECK-IMP:       # %bb.0:
; CHECK-IMP-NEXT:    ret
  ret void
}

; RUN: not llc -mtriple=riscv32 -mattr=+f -target-abi ilp32f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-UNIMP %s
; RUN: not llc -mtriple=riscv32 -mattr=+d -target-abi ilp32f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-UNIMP %s
; RUN: not llc -mtriple=riscv32 -mattr=+d -target-abi ilp32d < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-UNIMP %s
; RUN: not llc -mtriple=riscv32 -target-abi ilp32e < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-UNIMP %s
; RUN: not llc -mtriple=riscv64 -mattr=+f -target-abi lp64f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-UNIMP %s
; RUN: not llc -mtriple=riscv64 -mattr=+d -target-abi lp64f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-UNIMP %s
; RUN: not llc -mtriple=riscv64 -mattr=+d -target-abi lp64d < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-UNIMP %s

; CHECK-UNIMP: LLVM ERROR: Don't know how to lower this ABI
