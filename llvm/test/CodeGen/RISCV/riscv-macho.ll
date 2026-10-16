; RUN: llc -mtriple=riscv32-apple-macho %s -o - | FileCheck %s
; RUN: llc -mtriple=riscv32-apple-macho -filetype=obj %s -o %t.o
; RUN: llvm-objdump -D %t.o | FileCheck %s --check-prefix=CHECK-OBJ

; CHECK-LABEL: _main:
; CHECK: li a0, 0
; CHECK: ret

; CHECK-OBJ: li a0, 0
; CHECK-OBJ: ret
define i32 @main() nounwind {
  ret i32 0
}
