; RUN: llc -mtriple=riscv32 -mattr=+c  -filetype=obj < %s\
; RUN: | llvm-objdump -triple=riscv32 -mattr=+c -d -M no-aliases -\
; RUN: | FileCheck -check-prefix=CHECK %s

@ext = external global i32

define i32 @compress_test(i32 %a) {
; CHECK-LABEL: <compress_test>:
; CHECK:    c.add a0, a1
; CHECK-NEXT:    c.jr ra
  %1 = load i32, i32* @ext
  %2 = tail call i32 asm "add $0, $1, $2", "=r,r,r"(i32 %a, i32 %1)
  ret i32 %2
}

