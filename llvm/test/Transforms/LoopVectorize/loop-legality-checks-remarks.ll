; RUN: opt < %s -passes=loop-vectorize -pass-remarks=loop-vectorize -debug -disable-output 2>&1 | FileCheck %s
; REQUIRES: asserts

; Make sure LV legal bails out when the loop doesn't have a legal pre-header.
; CHECK-LABEL: 'not_exist_preheader'
; CHECK: LV: Not vectorizing: Loop doesn't have a legal pre-header.
define void @not_exist_preheader(ptr %dst, ptr %arg) nounwind uwtable {
entry:
  indirectbr ptr %arg, [label %exit.0, label %loop]

exit.0:
  ret void

loop:
  %iv = phi i32 [0, %entry], [%iv.next, %loop]
  %gep = getelementptr inbounds i32, ptr %dst, i32 %iv
  store i32 0, ptr %gep, align 4
  %iv.next = add i32 %iv, 1
  %cmp = icmp slt i32 %iv.next, 4
  br i1 %cmp, label %loop, label %exit

exit:
  ret void
}
