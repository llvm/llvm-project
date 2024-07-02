; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
;
; Verify we do not hoist I[c] without execution context because it
; is executed in a statement with an invalid domain and it depends
; on a parameter that was specialized by the domain.
;
; CHECK:         Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [c] -> { Stmt_if_then[i0] -> MemRef_I[-129] };
; CHECK-NEXT:            Execution Context: [c] -> {  : false }
; CHECK-NEXT:    }
;
; TODO: FIXME: We should remove the statement as it has an empty domain.
; CHECK:      Stmt_if_then
; CHECK-NEXT: Domain :=
; CHECK-NEXT: [c] -> { Stmt_if_then[i0] : false };
;
;    int I[1024];
;    void f(int *A, unsigned char c) {
;      for (int i = 0; i < 10; i++)
;        if ((signed char)(c + (unsigned char)1) == 127)
;          A[i] += I[c];
;        else
;          A[i] = 0;
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@I = common global [1024 x i32] zeroinitializer, align 16

define void @f(ptr %A, i8 zeroext %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 10
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add i8 %c, 1
  %cmp3 = icmp eq i8 %add, 128
  %arrayidx6 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  br i1 %cmp3, label %if.then, label %if.else

if.then:                                          ; preds = %for.body
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @I, i64 0, i8 %c
  %tmp = load i32, ptr %arrayidx, align 4
  %tmp1 = load i32, ptr %arrayidx6, align 4
  %add7 = add nsw i32 %tmp1, %tmp
  store i32 %add7, ptr %arrayidx6, align 4
  br label %for.inc

if.else:                                           ; preds = %if.then, %for.body
  store i32 0, ptr %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %if.else, if.then
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
