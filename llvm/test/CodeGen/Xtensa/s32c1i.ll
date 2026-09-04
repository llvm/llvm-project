; RUN: llc -mtriple=xtensa -mattr=+s32c1i  -filetype=obj %s -o - | llvm-objdump --arch=xtensa --mattr=s32c1i -d - | FileCheck %s -check-prefix=XTENSA

define i32 @constraint_i(i32 %a) {
; XTENSA: 0: 22 e2 01    s32c1i  a2, a2, 4
  %res = tail call i32 asm "s32c1i $0, $1, $2", "=r,r,i"(i32 %a, i32 4)
  ret i32 %res
}
