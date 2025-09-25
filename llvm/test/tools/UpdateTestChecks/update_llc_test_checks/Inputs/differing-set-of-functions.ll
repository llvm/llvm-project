; RUN: sed 's/FN/foo/g' %s | llc -mtriple=riscv32 \
; RUN:   | FileCheck -check-prefixes=CHECK,CHECKA %s
; RUN: sed 's/FN/foo/g' %s | llc -mtriple=riscv32 \
; RUN:   | FileCheck -check-prefixes=CHECK,CHECKB %s
; RUN: sed 's/FN/bar/g' %s | llc -mtriple=riscv32 \
; RUN:   | FileCheck -check-prefixes=CHECK,CHECKC %s

define i32 @FN() {
  ret i32 1
}

define i32 @common() {
  ret i32 100
}
