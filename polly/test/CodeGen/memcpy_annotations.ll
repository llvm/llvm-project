; RUN: opt %loadPolly -polly-codegen -S < %s | FileCheck %s
;
; Verify that @llvm.memcpy does not get a !alias.scope annotation.
; @llvm.memcpy takes two pointers, it is ambiguous to which the
; annotation applies.
;
; for (int j = 0; j < n; j += 1) {
;   memcpy(A, B, 8);
; }
;

declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32, i1)

define void @func(i32 %n, ptr noalias nonnull %A, ptr noalias nonnull %B) {
entry:
  br label %for

for:
  %j = phi i32 [0, %entry], [%j.inc, %inc]
  %j.cmp = icmp slt i32 %j, %n
  br i1 %j.cmp, label %body, label %exit

    body:
      call void @llvm.memcpy.p0.p0.i64(ptr nonnull %A, ptr %B, i64 8, i32 4, i1 false)
      br label %inc

inc:
  %j.inc = add nuw nsw i32 %j, 1
  br label %for

exit:
  br label %return

return:
  ret void
}


; CHECK-LABEL: polly.start:
; CHECK:         call void @llvm.memcpy
; CHECK-NOT:     !alias.scope
