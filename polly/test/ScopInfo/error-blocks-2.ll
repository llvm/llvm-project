; RUN: opt %loadPolly -polly-print-scops -disable-output \
; RUN: -polly-invariant-load-hoisting=true < %s | FileCheck %s
;
; CHECK:         Invariant Accesses: {
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, valid_val] -> { Stmt_for_body[i0] -> MemRef_valid[0] };
; CHECK-NEXT:            Execution Context: [N, valid_val] -> {  : N > 0 }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, valid_val] -> { Stmt_S[i0] -> MemRef_ptr_addr[0] };
; CHECK-NEXT:            Execution Context: [N, valid_val] -> { : }
; CHECK-NEXT:            ReadAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, valid_val] -> { Stmt_S[i0] -> MemRef_tmp2[0] };
; CHECK-NEXT:            Execution Context: [N, valid_val] -> { : N > 0 and (valid_val < 0 or valid_val > 0) }
; CHECK-NEXT:    }
; CHECK-NEXT:    Context:
; CHECK-NEXT:    [N, valid_val] -> {  : -2147483648 <= N <= 2147483647 and -2147483648 <= valid_val <= 2147483647 }
; CHECK-NEXT:    Assumed Context:
; CHECK-NEXT:    [N, valid_val] -> {  :  }
; CHECK-NEXT:    Invalid Context:
; CHECK-NEXT:    [N, valid_val] -> {  : valid_val = 0 and N > 0 }
;
; CHECK:         Statements {
; CHECK-NEXT:       Stmt_S
; CHECK-NEXT:            Domain :=
; CHECK-NEXT:                [N, valid_val] -> { Stmt_S[i0] : 0 <= i0 < N };
; CHECK-NEXT:            Schedule :=
; CHECK-NEXT:                [N, valid_val] -> { Stmt_S[i0] -> [i0] };
; CHECK-NEXT:            MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:                [N, valid_val] -> { Stmt_S[i0] -> MemRef_A[i0] };
; CHECK-NEXT:    }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @f(i32 %N, ptr noalias %valid, ptr noalias %ptr, ptr noalias %A) {
entry:
  %ptr.addr = alloca ptr, align 8
  store ptr %ptr, ptr %ptr.addr, align 8
  %tmp = sext i32 %N to i64
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %entry ]
  %cmp = icmp slt i64 %indvars.iv, %tmp
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %valid_val = load i32, ptr %valid, align 4
  %cmp1 = icmp eq i32 %valid_val, 0
  br i1 %cmp1, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  call void @doSth(ptr nonnull %ptr.addr)
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %S

S:                                                ; preds = %if.end
  %tmp2 = load ptr, ptr %ptr.addr, align 8
  %tmp3 = load i32, ptr %tmp2, align 4
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  store i32 %tmp3, ptr %arrayidx, align 4
  br label %for.inc

for.inc:                                          ; preds = %S
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

declare void @doSth(ptr)
