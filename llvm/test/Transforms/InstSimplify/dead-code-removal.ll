; RUN: opt -instsimplify -S < %s | FileCheck %s

define void @foo() nounwind {
  br i1 undef, label %1, label %4

; <label>:1                                       ; preds = %1, %0
; CHECK-NOT: phi
; CHECK-NOT: sub
  %2 = phi i32 [ %3, %1 ], [ undef, %0 ]
  %3 = sub i32 0, undef
  br label %1

; <label>:4                                       ; preds = %0
  ret void
}
