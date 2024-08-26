; RUN: opt %loadNPMPolly '-passes=print<polly-function-scops>' -polly-invariant-load-hoisting=true -disable-output < %s 2>&1 | FileCheck %s
;
; Verify we hoist I[0] without execution context even though it
; is executed in a statement with an invalid domain.
;
; CHECK:         Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [c] -> { Stmt_if_then[i0] -> MemRef_I[0] };
; CHECK-NEXT:            Execution Context: [c] -> {  :  }
; CHECK-NEXT:    }
;
;    int I[1];
;    void f(int *A, unsigned char c) {
;      for (int i = 0; i < 10; i++)
;        if ((signed char)(c + (unsigned char)1) > 0)
;          A[i] += I[0];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@I = common global [1 x i32] zeroinitializer, align 4

define void @f(ptr %A, i8 zeroext %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 10
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add i8 %c, 1
  %cmp3 = icmp sgt i8 %add, 0
  br i1 %cmp3, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %tmp = load i32, ptr @I, align 4
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %tmp1 = load i32, ptr %arrayidx, align 4
  %add5 = add nsw i32 %tmp1, %tmp
  store i32 %add5, ptr %arrayidx, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
