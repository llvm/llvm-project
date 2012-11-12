; RUN: llc < %s -mtriple=arm-apple-darwin -relocation-model=pic | FileCheck %s
; rdar://9027648

@A = available_externally hidden constant i32 1
@B = external hidden constant i32

define i32 @t1() {
  %tmp = load i32* @A
  store i32 %tmp, i32* @B
  ret i32 %tmp
}

; CHECK:      L_A$non_lazy_ptr:
; CHECK-NEXT: .long _A
; CHECK:      L_B$non_lazy_ptr:
; CHECK-NEXT: .long _B
