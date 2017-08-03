; RUN: llc -march=x86-64 < %s -disable-fp-elim | FileCheck %s

; This test is checking that we don't crash and we don't incorrectly fold
; a large displacement and a frame index into a single lea.
; <rdar://problem/9763308>

declare void @bar([39 x i8]*)
define i32 @f(i64 %a, i64 %b) nounwind readnone {
entry:
  %stack_main = alloca [39 x i8]
  call void @bar([39 x i8]* %stack_main)
  %tmp6 = add i64 %a, -2147483647
  %.sum = add i64 %tmp6, %b
  %tmp8 = getelementptr inbounds [39 x i8], [39 x i8]* %stack_main, i64 0, i64 %.sum
  %tmp9 = load i8, i8* %tmp8, align 1
  %tmp10 = sext i8 %tmp9 to i32
  ret i32 %tmp10
}
; CHECK-LABEL: f:
; CHECK: movsbl	-2147483647
