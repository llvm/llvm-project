; RUN: opt %loadPolly -polly-print-scops -polly-invariant-load-hoisting=true -disable-output < %s | FileCheck %s
;
; Verify we do hoist the invariant access to I with a execution context
; as the address computation might wrap in the original but not in our
; optimized version. For an input of c = 127 the original accessed address
; would be &I[-1] = &GI[128 -1] = &GI[127] but in our optimized version
; (due to the usage of i64 types) we would access
; &I[127 + 1] = &I[128] = &GI[256] which would here also be out-of-bounds.
;
; CHECK:        Invariant Accesses: {
; CHECK-NEXT:     ReadAccess := [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:       [c] -> { Stmt_for_body[i0] -> MemRef_GI[129 + c] };
; CHECK-NEXT:     Execution Context: [c] -> {  : c <= 126 }
; CHECK-NEXT:   }
;
;    int GI[256];
;    void f(int *A, unsigned char c) {
;      int *I = &GI[128];
;      for (int i = 0; i < 10; i++)
;        A[i] += I[(signed char)(c + (unsigned char)1)];
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@GI = common global [256 x i32] zeroinitializer, align 16

define void @f(i32* %A, i8 zeroext %c) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %exitcond = icmp ne i64 %indvars.iv, 10
  br i1 %exitcond, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add i8 %c, 1
  %idxprom = sext i8 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* getelementptr inbounds ([256 x i32], [256 x i32]* @GI, i64 0, i64 128), i64 %idxprom
  %tmp = load i32, i32* %arrayidx, align 4
  %arrayidx3 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %tmp1 = load i32, i32* %arrayidx3, align 4
  %add4 = add nsw i32 %tmp1, %tmp
  store i32 %add4, i32* %arrayidx3, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
