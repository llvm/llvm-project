; RUN: llc -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck %s

define i2 @f(i32 %arg) {
  %trunc = trunc i32 %arg to i1
  %sext = sext i1 %trunc to i2
  %or = or i2 %sext, 1
  ret i2 %or

; CHECK-LABEL: f:
; CHECK:      movl    %edi, %eax
; CHECK-NEXT: addb    %al, %al
; CHECK-NEXT: orb     $1, %al
; CHECK-NEXT: # kill
; CHECK-NEXT: retq
}
