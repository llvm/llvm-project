
; RUN: opt %loadPolly -polly-print-ast -polly-ignore-aliasing -disable-output < %s | FileCheck %s -check-prefix=SCALAR
; RUN: opt %loadPolly -polly-codegen -polly-ignore-aliasing -disable-output < %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; SCALAR: if (
; SCALAR:     {
; SCALAR:       Stmt_for_1(0);
; SCALAR:       for (int c0 = 1; c0 <= -Scalar0_val + 99; c0 += 1)
; SCALAR:         Stmt_for_1(c0);
; SCALAR:     }

; SCALAR: if (1)
; SCALAR:     Stmt_for_0(0);


define void @foo(i32* %A) {
entry:
  %Scalar0 = alloca i32
  br label %for.0

for.0:
  %Scalar0.val = load i32, i32* %Scalar0
  store i32 1, i32* %Scalar0
  br i1 false, label %for.0, label %for.1.preheader

for.1.preheader:
  fence seq_cst
  br label %for.1

for.1:
  %indvar.1 = phi i32 [ %Scalar0.val, %for.1.preheader ], [ %indvar.1.next, %for.1]
  %arrayidx.1 = getelementptr inbounds i32, i32* %A, i32 %indvar.1
  store i32 1, i32* %arrayidx.1
  %indvar.1.next = add nsw i32 %indvar.1, 1
  %cmp.1 = icmp slt i32 %indvar.1.next, 100
  br i1 %cmp.1, label %for.1, label %end

end:
  ret void
}
