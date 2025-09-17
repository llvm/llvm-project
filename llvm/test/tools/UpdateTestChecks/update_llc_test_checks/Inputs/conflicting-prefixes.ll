; RUN: sed 's/RETVAL/1/g' %s | llc -mtriple=riscv32 \
; RUN:   | FileCheck -check-prefixes=CHECK,CHECKA %s
; RUN: sed 's/RETVAL/2/g' %s | llc -mtriple=riscv32 \
; RUN:   | FileCheck -check-prefixes=CHECK,CHECKA %s
; RUN: sed 's/RETVAL/3/g' %s | llc -mtriple=riscv32 \
; RUN:   | FileCheck -check-prefixes=CHECK,CHECKB %s
; RUN: sed 's/RETVAL/4/g' %s | llc -mtriple=riscv32 \
; RUN:   | FileCheck -check-prefixes=CHECK,CHECKB %s

define i32 @foo() {
  ret i32 RETVAL
}

define i32 @bar() {
  ret i32 100
}
