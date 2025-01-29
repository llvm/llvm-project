; RUN: sed 's/SMALL_DATA_LIMIT/0/g' %s | \
; RUN:   llc -mtriple=riscv32 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-0 %s
; RUN: sed 's/SMALL_DATA_LIMIT/0/g' %s | \
; RUN:   llc -mtriple=riscv64 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-0 %s
; RUN: sed 's/SMALL_DATA_LIMIT/4/g' %s | \
; RUN:   llc -mtriple=riscv32 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-4 %s
; RUN: sed 's/SMALL_DATA_LIMIT/4/g' %s | \
; RUN:   llc -mtriple=riscv64 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-4 %s
; RUN: sed 's/SMALL_DATA_LIMIT/8/g' %s | \
; RUN:   llc -mtriple=riscv32 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-8 %s
; RUN: sed 's/SMALL_DATA_LIMIT/8/g' %s | \
; RUN:   llc -mtriple=riscv64 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-8 %s
; RUN: sed 's/SMALL_DATA_LIMIT/16/g' %s | \
; RUN:   llc -mtriple=riscv32 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-16 %s
; RUN: sed 's/SMALL_DATA_LIMIT/16/g' %s | \
; RUN:   llc -mtriple=riscv64 -mattr=+d | \
; RUN:   FileCheck -check-prefix=CHECK-SDL-16 %s

define dso_local float @foof() {
entry:
  ret float 0x400A08ACA0000000
}

define dso_local double @foo() {
entry:
  ret double 0x400A08AC91C3E242
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"SmallDataLimit", i32 SMALL_DATA_LIMIT}

; CHECK-SDL-0-NOT:    .section        .srodata.cst4
; CHECK-SDL-0-NOT:    .section        .srodata.cst8
; CHECK-SDL-4:        .section        .srodata.cst4
; CHECK-SDL-4-NOT:    .section        .srodata.cst8
; CHECK-SDL-8:        .section        .srodata.cst4
; CHECK-SDL-8:        .section        .srodata.cst8
; CHECK-SDL-16:       .section        .srodata.cst4
; CHECK-SDL-16:       .section        .srodata.cst8
