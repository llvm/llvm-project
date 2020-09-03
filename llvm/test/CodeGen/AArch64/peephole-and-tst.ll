; RUN: llc < %s -mtriple=aarch64-- | FileCheck %s

%struct.anon = type { i32*, i32* }

@ptr_wrapper = common dso_local local_unnamed_addr global %struct.anon* null, align 8

define dso_local i32 @test_func_i32_two_uses(i32 %in, i32 %bit, i32 %mask) local_unnamed_addr {
entry:
  %0 = load %struct.anon*, %struct.anon** @ptr_wrapper, align 8
  %result = getelementptr inbounds %struct.anon, %struct.anon* %0, i64 0, i32 1
  %tobool2 = icmp ne i32 %mask, 0
  br label %do.body

do.body:                                          ; preds = %4, %entry
; CHECK-LABEL: test_func_i32_two_uses:
; CHECK: ands [[DSTREG:w[0-9]+]]
; Usage #1
; CHECK: cmp [[DSTREG]]
; Usage #2
; CHECK: cbz [[DSTREG]]
  %bit.addr.0 = phi i32 [ %bit, %entry ], [ %shl, %4 ]
  %retval1.0 = phi i32 [ 0, %entry ], [ %retval1.1, %4 ]
  %and = and i32 %bit.addr.0, %in
  %tobool = icmp eq i32 %and, 0
  %not.tobool = xor i1 %tobool, true
  %inc = zext i1 %not.tobool to i32
  %retval1.1 = add nuw nsw i32 %retval1.0, %inc
  %1 = xor i1 %tobool, true
  %2 = or i1 %tobool2, %1
  %dummy = and i32 %mask, %in
  %use_and = icmp eq i32 %and, %dummy
  %dummy_or = or i1 %use_and, %2
  br i1 %dummy_or, label %3, label %4

3:                                                ; preds = %do.body
  store i32* null, i32** %result, align 8
  br label %4

4:                                                ; preds = %do.body, %3
  %shl = shl i32 %bit.addr.0, 1
  %tobool6 = icmp eq i32 %shl, 0
  br i1 %tobool6, label %do.end, label %do.body

do.end:                                           ; preds = %4
  ret i32 %retval1.1
}

define dso_local i32 @test_func_i64_one_use(i64 %in, i64 %bit, i64 %mask) local_unnamed_addr #0 {
entry:
  %0 = load %struct.anon*, %struct.anon** @ptr_wrapper, align 8
  %result = getelementptr inbounds %struct.anon, %struct.anon* %0, i64 0, i32 1
  %tobool2 = icmp ne i64 %mask, 0
  br label %do.body

do.body:                                          ; preds = %4, %entry
; CHECK-LABEL: test_func_i64_one_use:
; CHECK: ands [[DSTREG:x[0-9]+]], [[SRCREG1:x[0-9]+]], [[SRCREG2:x[0-9]+]]
; CHECK-NEXT: orr [[DSTREG]], [[SRCREG_ORR:x[0-9]+]], [[DSTREG]]
  %bit.addr.0 = phi i64 [ %bit, %entry ], [ %shl, %4 ]
  %retval1.0 = phi i32 [ 0, %entry ], [ %retval1.1, %4 ]
  %and = and i64 %bit.addr.0, %in
  %tobool = icmp eq i64 %and, 0
  %not.tobool = xor i1 %tobool, true
  %inc = zext i1 %not.tobool to i32
  %retval1.1 = add nuw nsw i32 %retval1.0, %inc
  %1 = xor i1 %tobool, true
  %2 = or i1 %tobool2, %1
  br i1 %2, label %3, label %4

3:                                                ; preds = %do.body
  store i32* null, i32** %result, align 8
  br label %4

4:                                                ; preds = %do.body, %3
  %shl = shl i64 %bit.addr.0, 1
  %tobool6 = icmp eq i64 %shl, 0
  br i1 %tobool6, label %do.end, label %do.body

do.end:                                           ; preds = %4
  ret i32 %retval1.1
}
