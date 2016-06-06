; RUN: opt %loadPolly -analyze -polly-scops < %s | FileCheck %s
; RUN: opt %loadPolly -S -polly-codegen < %s | FileCheck %s --check-prefix=IR
; RUN: opt %loadPolly -S -polly-codegen --polly-overflow-tracking=always < %s | FileCheck %s --check-prefix=IRA
;
; As (p + q) can overflow we have to check that we load from
; I[p + q] only if it does not.
;
; CHECK:         Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, p, q] -> { Stmt_for_body[i0] -> MemRef_I[p + q] };
; CHECK-NEXT:            Execution Context: [N, p, q] -> {  : N > 0 and -2147483648 - p <= q <= 2147483647 - p }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, p, q] -> { Stmt_for_body[i0] -> MemRef_tmp1[0] };
; CHECK-NEXT:            Execution Context: [N, p, q] -> {  : N > 0 }
; CHECK-NEXT:    }
;
; IR:  polly.preload.merge:
; IR-NEXT:   %polly.preload.tmp1.merge = phi i32* [ %polly.access.I.load, %polly.preload.exec ], [ null, %polly.preload.cond ]
; IR-NEXT:   store i32* %polly.preload.tmp1.merge, i32** %tmp1.preload.s2a
; IR-NEXT:   %11 = icmp sge i32 %N, 1
; IR-NEXT:   %12 = sext i32 %p to i33
; IR-NEXT:   %13 = sext i32 %q to i33
; IR-NEXT:   %14 = add nsw i33 %12, %13
; IR-NEXT:   %15 = icmp sle i33 %14, 2147483647
; IR-NEXT:   %16 = and i1 %11, %15
; IR-NEXT:   %17 = sext i32 %p to i33
; IR-NEXT:   %18 = sext i32 %q to i33
; IR-NEXT:   %19 = add nsw i33 %17, %18
; IR-NEXT:   %20 = icmp sge i33 %19, -2147483648
; IR-NEXT:   %21 = and i1 %16, %20
; IR-NEXT:   br label %polly.preload.cond1
;
; IR:       polly.preload.cond1:
; IR-NEXT:    br i1 %21, label %polly.preload.exec3, label %polly.preload.merge2

; IR:      polly.preload.exec3:
; IR-NEXT:   %polly.access.polly.preload.tmp1.merge = getelementptr i32, i32* %polly.preload.tmp1.merge, i1 false
; IR-NEXT:   %polly.access.polly.preload.tmp1.merge.load = load i32, i32* %polly.access.polly.preload.tmp1.merge, align 4
;
; IRA:      polly.preload.merge:
; IRA-NEXT:   %polly.preload.tmp1.merge = phi i32* [ %polly.access.I.load, %polly.preload.exec ], [ null, %polly.preload.cond ]
; IRA-NEXT:   store i32* %polly.preload.tmp1.merge, i32** %tmp1.preload.s2a
; IRA-NEXT:   %11 = icmp sge i32 %N, 1
; IRA-NEXT:   %12 = sext i32 %p to i33
; IRA-NEXT:   %13 = sext i32 %q to i33
; IRA-NEXT:   %14 = call { i33, i1 } @llvm.sadd.with.overflow.i33(i33 %12, i33 %13)
; IRA-NEXT:   %.obit5 = extractvalue { i33, i1 } %14, 1
; IRA-NEXT:   %.res6 = extractvalue { i33, i1 } %14, 0
; IRA-NEXT:   %15 = icmp sle i33 %.res6, 2147483647
; IRA-NEXT:   %16 = and i1 %11, %15
; IRA-NEXT:   %17 = sext i32 %p to i33
; IRA-NEXT:   %18 = sext i32 %q to i33
; IRA-NEXT:   %19 = call { i33, i1 } @llvm.sadd.with.overflow.i33(i33 %17, i33 %18)
; IRA-NEXT:   %.obit7 = extractvalue { i33, i1 } %19, 1
; IRA-NEXT:   %.res8 = extractvalue { i33, i1 } %19, 0
; IRA-NEXT:   %20 = icmp sge i33 %.res8, -2147483648
; IRA-NEXT:   %21 = and i1 %16, %20
; IRA-NEXT:   %polly.preload.cond.overflown9 = xor i1 %.obit7, true
; IRA-NEXT:   %polly.preload.cond.result10 = and i1 %21, %polly.preload.cond.overflown9
; IRA-NEXT:   br label %polly.preload.cond11
;
; IRA:      polly.preload.cond11:
; IRA-NEXT:   br i1 %polly.preload.cond.result10
;
; IRA:      polly.preload.exec13:
; IRA-NEXT:   %polly.access.polly.preload.tmp1.merge = getelementptr i32, i32* %polly.preload.tmp1.merge, i1 false
; IRA-NEXT:   %polly.access.polly.preload.tmp1.merge.load = load i32, i32* %polly.access.polly.preload.tmp1.merge, align 4
;
;    void f(int **I, int *A, int N, int p, int q) {
;      for (int i = 0; i < N; i++)
;        A[i] = *(I[p + q]);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32** %I, i32* %A, i32 %N, i32 %p, i32 %q) {
entry:
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %add = add i32 %p, %q
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32*, i32** %I, i64 %idxprom
  %tmp1 = load i32*, i32** %arrayidx, align 8
  %tmp2 = load i32, i32* %tmp1, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  store i32 %tmp2, i32* %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
