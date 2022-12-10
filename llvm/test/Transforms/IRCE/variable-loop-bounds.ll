; RUN: opt -irce -S -verify-loop-info -irce-print-changed-loops -irce-skip-profitability-checks < %s 2>&1 | FileCheck %s

; CHECK: irce: in function test_inc_eq: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>
; CHECK: irce: in function test_inc_ne: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>
; CHECK: irce: in function test_inc_slt: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>
; CHECK: irce: in function test_inc_ult: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%if.then,%for.inc<latch><exiting>
; CHECK: irce: in function signed_var_imm_dec_sgt: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%for.inc<latch><exiting>
; CHECK-NOT: irce: in function signed_var_imm_dec_slt: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%for.inc<latch><exiting>
; CHECK: irce: in function signed_var_imm_dec_sge: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%for.inc<latch><exiting>
; CHECK: irce: in function signed_var_imm_dec_ne: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%for.inc<latch><exiting>
; CHECK-NOT: irce: in function signed_var_imm_dec_eq: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%for.inc<latch><exiting>
; CHECK-NOT: irce: in function test_dec_bound_with_smaller_start_than_bound: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%for.dec<latch><exiting>
; CHECK-NOT: irce: in function test_inc_bound_with_bigger_start_than_bound: constrained Loop at depth 1 containing: %for.body<header>,%if.else,%for.dec<latch><exiting>

; CHECK-LABEL: test_inc_eq(
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp ult i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_eq(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.017
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %i.017
  %1 = load i32, ptr %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %i.017
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, ptr %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, ptr %a, i32 %i.017
  store i32 %add6, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp eq i32 %inc, %N
  br i1 %exitcond, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: test_inc_ne
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp ult i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_ne(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.017
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %i.017
  %1 = load i32, ptr %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %i.017
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, ptr %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, ptr %a, i32 %i.017
  store i32 %add6, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp ne i32 %inc, %N
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: test_inc_slt(
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp slt i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_slt(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp sgt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.017
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %i.017
  %1 = load i32, ptr %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %i.017
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, ptr %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, ptr %a, i32 %i.017
  store i32 %add6, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp slt i32 %inc, %N
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: test_inc_ult
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %inc, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp ult i32 [[PSEUDO_PHI]], %N
; CHECK: br i1 [[COND]], label %main.pseudo.exit, label %for.cond.cleanup.loopexit
define void @test_inc_ult(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %N) {
entry:
  %cmp16 = icmp ugt i32 %N, 0
  br i1 %cmp16, label %for.body, label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %i.017 = phi i32 [ %inc, %for.inc ], [ 0, %entry ]
  %cmp1 = icmp ult i32 %i.017, 512
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %i.017
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %i.017
  %1 = load i32, ptr %arrayidx2, align 4
  br i1 %cmp1, label %if.then, label %if.else

if.then:
  %sub = sub i32 %0, %1
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %i.017
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %sub, %2
  store i32 %add, ptr %arrayidx3, align 4
  br label %for.inc

if.else:
  %add6 = add nsw i32 %1, %0
  %arrayidx7 = getelementptr inbounds i32, ptr %a, i32 %i.017
  store i32 %add6, ptr %arrayidx7, align 4
  br label %for.inc

for.inc:
  %inc = add nuw nsw i32 %i.017, 1
  %exitcond = icmp ult i32 %inc, %N
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: signed_var_imm_dec_sgt(
; CHECK: main.exit.selector:
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %dec, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp sgt i32 [[PSEUDO_PHI]], %M
; CHECK: br i1 [[COND]]
define void @signed_var_imm_dec_sgt(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %M) {
entry:
  %cmp14 = icmp slt i32 %M, 1024
  br i1 %cmp14, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %iv = phi i32 [ %dec, %for.inc ], [ 1024, %entry ]
  %cmp1 = icmp slt i32 %iv, 1024
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %iv
  br i1 %cmp1, label %for.inc, label %if.else

if.else:                                          ; preds = %for.body
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %2, %mul
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.else
  %storemerge = phi i32 [ %add, %if.else ], [ %mul, %for.body ]
  store i32 %storemerge, ptr %arrayidx3, align 4
  %dec = add nsw i32 %iv, -1
  %cmp = icmp sgt i32 %dec, %M
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

; CHECK-LABEL: signed_var_imm_dec_sge(
; CHECK: main.exit.selector:          ; preds = %for.inc
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %iv, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp sgt i32 [[PSEUDO_PHI]], %M
; CHECK: br i1 [[COND]]
define void @signed_var_imm_dec_sge(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %M) {
entry:
  %cmp14 = icmp sgt i32 %M, 1024
  br i1 %cmp14, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %iv = phi i32 [ %dec, %for.inc ], [ 1024, %entry ]
  %cmp1 = icmp slt i32 %iv, 1024
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %iv
  br i1 %cmp1, label %for.inc, label %if.else

if.else:                                          ; preds = %for.body
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %2, %mul
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.else
  %storemerge = phi i32 [ %add, %if.else ], [ %mul, %for.body ]
  store i32 %storemerge, ptr %arrayidx3, align 4
  %dec = add nsw i32 %iv, -1
  %cmp = icmp sgt i32 %iv, %M
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @signed_var_imm_dec_slt(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %M) {
entry:
  %cmp14 = icmp sgt i32 %M, 1024
  br i1 %cmp14, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %iv = phi i32 [ %dec, %for.inc ], [ 1024, %entry ]
  %cmp1 = icmp slt i32 %iv, 1024
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %iv
  br i1 %cmp1, label %for.inc, label %if.else

if.else:                                          ; preds = %for.body
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %2, %mul
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.else
  %storemerge = phi i32 [ %add, %if.else ], [ %mul, %for.body ]
  store i32 %storemerge, ptr %arrayidx3, align 4
  %dec = add nsw i32 %iv, -1
  %cmp = icmp slt i32 %iv, %M
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: signed_var_imm_dec_ne(
; CHECK: main.exit.selector:          ; preds = %for.inc
; CHECK: [[PSEUDO_PHI:%[^ ]+]] = phi i32 [ %dec, %for.inc ]
; CHECK: [[COND:%[^ ]+]] = icmp sgt i32 [[PSEUDO_PHI]], %M
; CHECK: br i1 [[COND]]
define void @signed_var_imm_dec_ne(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %M) {
entry:
  %cmp14 = icmp slt i32 %M, 1024
  br i1 %cmp14, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %iv = phi i32 [ %dec, %for.inc ], [ 1024, %entry ]
  %cmp1 = icmp slt i32 %iv, 1024
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %iv
  br i1 %cmp1, label %for.inc, label %if.else

if.else:                                          ; preds = %for.body
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %2, %mul
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.else
  %storemerge = phi i32 [ %add, %if.else ], [ %mul, %for.body ]
  store i32 %storemerge, ptr %arrayidx3, align 4
  %dec = add nsw i32 %iv, -1
  %cmp = icmp ne i32 %dec, %M
  br i1 %cmp, label %for.body, label %for.cond.cleanup
}

define void @signed_var_imm_dec_eq(ptr nocapture %a, ptr nocapture readonly %b, ptr nocapture readonly %c, i32 %M) {
entry:
  %cmp14 = icmp slt i32 %M, 1024
  br i1 %cmp14, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.inc, %entry
  ret void

for.body:                                         ; preds = %entry, %for.inc
  %iv = phi i32 [ %dec, %for.inc ], [ 1024, %entry ]
  %cmp1 = icmp slt i32 %iv, 1024
  %arrayidx = getelementptr inbounds i32, ptr %b, i32 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr %c, i32 %iv
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul nsw i32 %1, %0
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i32 %iv
  br i1 %cmp1, label %for.inc, label %if.else

if.else:                                          ; preds = %for.body
  %2 = load i32, ptr %arrayidx3, align 4
  %add = add nsw i32 %2, %mul
  br label %for.inc

for.inc:                                          ; preds = %for.body, %if.else
  %storemerge = phi i32 [ %add, %if.else ], [ %mul, %for.body ]
  store i32 %storemerge, ptr %arrayidx3, align 4
  %dec = add nsw i32 %iv, -1
  %cmp = icmp eq i32 %dec, %M
  br i1 %cmp, label %for.cond.cleanup, label %for.body
}

; CHECK-LABEL: @test_dec_bound_with_smaller_start_than_bound(
; CHECK-NOT:       preloop.exit.selector:
define void @test_dec_bound_with_smaller_start_than_bound(i64 %0) {
entry:
  br label %for.body

for.body:                                                ; preds = %for.dec, %entry
  %iv = phi i64 [ %dec, %for.dec ], [ 0, %entry ]
  %1 = icmp slt i64 %iv, %0
  br i1 %1, label %if.else, label %for.dec

if.else:                                                ; preds = %for.body
  br label %for.dec

for.dec:                                                ; preds = %if.else, %for.body
  %dec = sub nuw nsw i64 %iv, 1
  %2 = icmp slt i64 %dec, 1
  br i1 %2, label %exit, label %for.body

exit:                                               ; preds = %for.dec
  ret void
}

; CHECK-LABEL: @test_inc_bound_with_bigger_start_than_bound(
; CHECK-NOT:       main.exit.selector:
define void @test_inc_bound_with_bigger_start_than_bound(i32 %0) {
entry:
  br label %for.body

for.body:                                                ; preds = %for.inc, %entry
  %iv = phi i32 [ %inc, %for.inc ], [ 200, %entry ]
  %1 = icmp slt i32 %iv, %0
  br i1 %1, label %if.else, label %for.inc

if.else:                                                ; preds = %for.body
  br label %for.inc

for.inc:                                                ; preds = %if.else, %for.body
  %inc = add nsw i32 %iv, 1
  %2 = icmp sgt i32 %inc, 100
  br i1 %2, label %exit, label %for.body

exit:                                                ; preds = %for.inc
  ret void
}
