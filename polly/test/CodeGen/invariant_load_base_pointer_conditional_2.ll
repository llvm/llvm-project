; RUN: opt %loadPolly -polly-invariant-load-hoisting=true -polly-print-scops -disable-output < %s | FileCheck %s
; RUN: opt %loadPolly -S -polly-codegen -polly-invariant-load-hoisting=true < %s | FileCheck %s --check-prefix=IR
; RUN: opt %loadPolly -S -polly-codegen -polly-invariant-load-hoisting=true --polly-overflow-tracking=always < %s | FileCheck %s --check-prefix=IRA
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
; IR:      polly.preload.merge:
; IR-NEXT:   %polly.preload.tmp1.merge = phi ptr [ %polly.access.I.load, %polly.preload.exec ], [ null, %polly.preload.cond ]
; IR-NEXT:   store ptr %polly.preload.tmp1.merge, ptr %tmp1.preload.s2a
; IR-NEXT:   %12 = sext i32 %N to i64
; IR-NEXT:   %13 = icmp sge i64 %12, 1
; IR-NEXT:   %14 = sext i32 %q to i64
; IR-NEXT:   %15 = sext i32 %p to i64
; IR-NEXT:   %16 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %15, i64 %14)
; IR-NEXT:   %.obit4 = extractvalue { i64, i1 } %16, 1
; IR-NEXT:   %polly.overflow.state5 = or i1 false, %.obit4
; IR-NEXT:   %.res6 = extractvalue { i64, i1 } %16, 0
; IR-NEXT:   %17 = icmp sle i64 %.res6, 2147483647
; IR-NEXT:   %18 = and i1 %13, %17
; IR-NEXT:   %19 = sext i32 %q to i64
; IR-NEXT:   %20 = sext i32 %p to i64
; IR-NEXT:   %21 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %20, i64 %19)
; IR-NEXT:   %.obit7 = extractvalue { i64, i1 } %21, 1
; IR-NEXT:   %polly.overflow.state8 = or i1 %polly.overflow.state5, %.obit7
; IR-NEXT:   %.res9 = extractvalue { i64, i1 } %21, 0
; IR-NEXT:   %22 = icmp sge i64 %.res9, -2147483648
; IR-NEXT:   %23 = and i1 %18, %22
; IR-NEXT:   %polly.preload.cond.overflown10 = xor i1 %polly.overflow.state8, true
; IR-NEXT:   %polly.preload.cond.result11 = and i1 %23, %polly.preload.cond.overflown10
; IR-NEXT:   br label %polly.preload.cond12
;
; IR:      polly.preload.cond12:
; IR-NEXT:   br i1 %polly.preload.cond.result11
;
; IR:      polly.preload.exec14:
; IR-NEXT:   %polly.access.polly.preload.tmp1.merge = getelementptr i32, ptr %polly.preload.tmp1.merge, i64 0
; IR-NEXT:   %polly.access.polly.preload.tmp1.merge.load = load i32, ptr %polly.access.polly.preload.tmp1.merge, align 4
;
; IRA:      polly.preload.merge:
; IRA-NEXT:   %polly.preload.tmp1.merge = phi ptr [ %polly.access.I.load, %polly.preload.exec ], [ null, %polly.preload.cond ]
; IRA-NEXT:   store ptr %polly.preload.tmp1.merge, ptr %tmp1.preload.s2a
; IRA-NEXT:   %12 = sext i32 %N to i64
; IRA-NEXT:   %13 = icmp sge i64 %12, 1
; IRA-NEXT:   %14 = sext i32 %q to i64
; IRA-NEXT:   %15 = sext i32 %p to i64
; IRA-NEXT:   %16 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %15, i64 %14)
; IRA-NEXT:   %.obit5 = extractvalue { i64, i1 } %16, 1
; IRA-NEXT:   %.res6 = extractvalue { i64, i1 } %16, 0
; IRA-NEXT:   %17 = icmp sle i64 %.res6, 2147483647
; IRA-NEXT:   %18 = and i1 %13, %17
; IRA-NEXT:   %19 = sext i32 %q to i64
; IRA-NEXT:   %20 = sext i32 %p to i64
; IRA-NEXT:   %21 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 %20, i64 %19)
; IRA-NEXT:   %.obit7 = extractvalue { i64, i1 } %21, 1
; IRA-NEXT:   %.res8 = extractvalue { i64, i1 } %21, 0
; IRA-NEXT:   %22 = icmp sge i64 %.res8, -2147483648
; IRA-NEXT:   %23 = and i1 %18, %22
; IRA-NEXT:   %polly.preload.cond.overflown9 = xor i1 %.obit7, true
; IRA-NEXT:   %polly.preload.cond.result10 = and i1 %23, %polly.preload.cond.overflown9
; IRA-NEXT:   br label %polly.preload.cond11
;
; IRA:      polly.preload.cond11:
; IRA-NEXT:   br i1 %polly.preload.cond.result10
;
; IRA:      polly.preload.exec13:
; IRA-NEXT:   %polly.access.polly.preload.tmp1.merge = getelementptr i32, ptr %polly.preload.tmp1.merge, i64 0
; IRA-NEXT:   %polly.access.polly.preload.tmp1.merge.load = load i32, ptr %polly.access.polly.preload.tmp1.merge, align 4
;
;    void f(int **I, int *A, int N, int p, int q) {
;      for (int i = 0; i < N; i++)
;        A[i] = *(I[p + q]);
;    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(ptr %I, ptr %A, i32 %N, i32 %p, i32 %q) {
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
  %arrayidx = getelementptr inbounds ptr, ptr %I, i64 %idxprom
  %tmp1 = load ptr, ptr %arrayidx, align 8
  %tmp2 = load i32, ptr %tmp1, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %tmp2, ptr %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
