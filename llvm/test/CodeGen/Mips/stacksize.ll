; RUN: llc -mtriple=mipsel -relocation-model=static < %s | FileCheck %s 

define i32 @foo(i32 %a) nounwind readnone {
entry:
; check that stack size is zero.
; CHECK-NOT: addiu $sp, $sp
  %add = add nsw i32 %a, 1
  ret i32 %add
}
