; RUN: opt -early-cse -S < %s | FileCheck %s
; PR12231

declare i32 @f()

define i32 @fn() {
entry:
  br label %lbl_1215

lbl_1215:
  %ins34 = phi i32 [ %ins35, %xxx ], [ undef, %entry ]
  ret i32 %ins34

xxx:
  %ins35 = call i32 @f()
  br label %lbl_1215
}

; CHECK: define i32 @fn
