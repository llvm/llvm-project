; RUN: llc -mtriple=riscv32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -target-abi ilp32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -target-abi ilp32e < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -mattr=+f -target-abi ilp32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -mattr=+d -target-abi ilp32 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -target-abi lp64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -target-abi lp64e < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+f -target-abi lp64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64 < %s \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -mattr=+f -target-abi ilp32f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -mattr=+d -target-abi ilp32f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv32 -mattr=+d -target-abi ilp32d < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+f -target-abi lp64f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64f < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64d < %s 2>&1 \
; RUN:   | FileCheck -check-prefix=CHECK-IMP %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64 -filetype=obj < %s \
; RUN:   | llvm-readelf -h - | FileCheck -check-prefix=CHECK-OBJ-LP64 %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64f -filetype=obj < %s \
; RUN:   | llvm-readelf -h - | FileCheck -check-prefix=CHECK-OBJ-LP64F %s
; RUN: llc -mtriple=riscv64 -mattr=+d -target-abi lp64d -filetype=obj < %s \
; RUN:   | llvm-readelf -h - | FileCheck -check-prefix=CHECK-OBJ-LP64D %s
; RUN: llc -mtriple=riscv64 -target-abi lp64e -filetype=obj < %s \
; RUN:   | llvm-readelf -h - | FileCheck -check-prefix=CHECK-OBJ-LP64E %s

; CHECK-OBJ-LP64:  Flags: 0x0{{$}}
; CHECK-OBJ-LP64F: Flags: 0x2, single-float ABI{{$}}
; CHECK-OBJ-LP64D: Flags: 0x4, double-float ABI{{$}}
; CHECK-OBJ-LP64E: Flags: 0x8, RVE{{$}}

define void @nothing() nounwind {
; CHECK-IMP-LABEL: nothing:
; CHECK-IMP:       # %bb.0:
; CHECK-IMP-NEXT:    ret
  ret void
}
