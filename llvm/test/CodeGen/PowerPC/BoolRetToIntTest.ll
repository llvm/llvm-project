; RUN: opt -bool-ret-to-int -S -o - < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; CHECK-LABEL: notBoolRet
define signext i32 @notBoolRet() {
entry:
; CHECK: ret i32 1
  ret i32 1
}

; CHECK-LABEL: find
define zeroext i1 @find(i8** readonly %begin, i8** readnone %end, i1 (i8*)* nocapture %hasProp) {
entry:
  %cmp.4 = icmp eq i8** %begin, %end
  br i1 %cmp.4, label %cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond:                                         ; preds = %for.body
  %cmp = icmp eq i8** %incdec.ptr, %end
  br i1 %cmp, label %cleanup.loopexit, label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.cond
  %curr.05 = phi i8** [ %incdec.ptr, %for.cond ], [ %begin, %for.body.preheader ]
  %0 = load i8*, i8** %curr.05, align 8
  %call = tail call zeroext i1 %hasProp(i8* %0)
  %incdec.ptr = getelementptr inbounds i8*, i8** %curr.05, i64 1
  br i1 %call, label %cleanup.loopexit, label %for.cond

cleanup.loopexit:                                 ; preds = %for.body, %for.cond
; CHECK: [[PHI:%.+]] = phi i64 [ 1, %for.body ], [ 0, %for.cond ]
  %cleanup.dest.slot.0.ph = phi i1 [ true, %for.body ], [ false, %for.cond ]
  br label %cleanup

cleanup:                                          ; preds = %cleanup.loopexit, %entry
; CHECK: = phi i64 [ 0, %entry ], [ [[PHI]], %cleanup.loopexit ]
  %cleanup.dest.slot.0 = phi i1 [ false, %entry ], [ %cleanup.dest.slot.0.ph, %cleanup.loopexit ]
; CHECK: [[REG:%.+]] = trunc i64 {{%.+}} to i1
; CHECK: ret i1 [[REG]]
  ret i1 %cleanup.dest.slot.0
}

; CHECK-LABEL: retFalse
define zeroext i1 @retFalse() {
entry:
; CHECK: ret i1 false
  ret i1 false
}

; CHECK-LABEL: retCvtFalse
define zeroext i1 @retCvtFalse() {
entry:
; CHECK: ret i1 false
  ret i1 trunc(i32 0 to i1)
}

; CHECK-LABEL: find_cont
define void @find_cont(i8** readonly %begin, i8** readnone %end, i1 (i8*)* nocapture %hasProp, void (i1)* nocapture %cont) {
entry:
  %cmp.4 = icmp eq i8** %begin, %end
  br i1 %cmp.4, label %cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond:                                         ; preds = %for.body
  %cmp = icmp eq i8** %incdec.ptr, %end
  br i1 %cmp, label %cleanup.loopexit, label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.cond
  %curr.05 = phi i8** [ %incdec.ptr, %for.cond ], [ %begin, %for.body.preheader ]
  %0 = load i8*, i8** %curr.05, align 8
  %call = tail call zeroext i1 %hasProp(i8* %0)
  %incdec.ptr = getelementptr inbounds i8*, i8** %curr.05, i64 1
  br i1 %call, label %cleanup.loopexit, label %for.cond

cleanup.loopexit:                                 ; preds = %for.body, %for.cond
; CHECK: [[PHI:%.+]] = phi i64 [ 1, %for.body ], [ 0, %for.cond ]
  %cleanup.dest.slot.0.ph = phi i1 [ true, %for.body ], [ false, %for.cond ]
  br label %cleanup

cleanup:                                          ; preds = %cleanup.loopexit, %entry
; CHECK: = phi i64 [ 0, %entry ], [ [[PHI]], %cleanup.loopexit ]
  %cleanup.dest.slot.0 = phi i1 [ false, %entry ], [ %cleanup.dest.slot.0.ph, %cleanup.loopexit ]
; CHECK: [[REG:%.+]] = trunc i64 {{%.+}} to i1
; CHECK: call void %cont(i1 [[REG]]
  tail call void %cont(i1 %cleanup.dest.slot.0)
  ret void
}

; CHECK-LABEL: find_cont_ret
define zeroext i1 @find_cont_ret(i8** readonly %begin, i8** readnone %end, i1 (i8*)* nocapture %hasProp, void (i1)* nocapture %cont) {
entry:
  %cmp.4 = icmp eq i8** %begin, %end
  br i1 %cmp.4, label %cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond:                                         ; preds = %for.body
  %cmp = icmp eq i8** %incdec.ptr, %end
  br i1 %cmp, label %cleanup.loopexit, label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.cond
  %curr.05 = phi i8** [ %incdec.ptr, %for.cond ], [ %begin, %for.body.preheader ]
  %0 = load i8*, i8** %curr.05, align 8
  %call = tail call zeroext i1 %hasProp(i8* %0)
  %incdec.ptr = getelementptr inbounds i8*, i8** %curr.05, i64 1
  br i1 %call, label %cleanup.loopexit, label %for.cond

cleanup.loopexit:                                 ; preds = %for.body, %for.cond
; CHECK: [[PHI:%.+]] = phi i64 [ 1, %for.body ], [ 0, %for.cond ]
  %cleanup.dest.slot.0.ph = phi i1 [ true, %for.body ], [ false, %for.cond ]
  br label %cleanup

cleanup:                                          ; preds = %cleanup.loopexit, %entry
; CHECK: = phi i64 [ 0, %entry ], [ [[PHI]], %cleanup.loopexit ]
  %cleanup.dest.slot.0 = phi i1 [ false, %entry ], [ %cleanup.dest.slot.0.ph, %cleanup.loopexit ]
; CHECK: [[REG:%.+]] = trunc i64 {{%.+}} to i1
; CHECK: call void %cont(i1 [[REG]]
  tail call void %cont(i1 %cleanup.dest.slot.0)
; CHECK: [[REG:%.+]] = trunc i64 {{%.+}} to i1
; CHECK: ret i1 [[REG]]
  ret i1 %cleanup.dest.slot.0
}

; CHECK-LABEL: arg_operand
define zeroext i1 @arg_operand(i1 %operand) {
entry:
  br i1 %operand, label %foo, label %cleanup

foo:
  br label %cleanup

cleanup:
; CHECK: [[REG:%.+]] = trunc i64 {{%.+}} to i1
; CHECK: ret i1 [[REG]]
  %result = phi i1 [ false, %foo ], [ %operand, %entry ]
  ret i1 %result
}

; CHECK-LABEL: bad_use
define zeroext i1 @bad_use(i1 %operand) {
entry:
  br i1 %operand, label %foo, label %cleanup

foo:
  br label %cleanup

cleanup:
; CHECK: [[REG:%.+]] = phi i1
; CHECK: ret i1 [[REG]]
  %result = phi i1 [ false, %foo], [ true, %entry ]
  %0 = icmp eq i1 %result, %operand
  ret i1 %result
}

; CHECK-LABEL: bad_use_closure
define zeroext i1 @bad_use_closure(i1 %operand) {
entry:
  br i1 %operand, label %foo, label %cleanup

foo:
  %bar = phi i1 [ false, %entry ]
  %0 = icmp eq i1 %bar, %operand
  br label %cleanup

cleanup:
; CHECK: [[REG:%.+]] = phi i1 [ true
; CHECK: ret i1 [[REG]]
  %result = phi i1 [ true, %entry ], [ %bar, %foo]
  ret i1 %result
}

; CHECK-LABEL: arg_test
define zeroext i1 @arg_test(i1 %operand) {
entry:
  br i1 %operand, label %foo, label %cleanup

foo:
  %bar = phi i1 [ false, %entry ]
  br label %cleanup

; CHECK-LABEL: cleanup
cleanup:
; CHECK: [[REG:%.+]] = trunc i64 {{%.+}} to i1
; CHECK: ret i1 [[REG]]
  %result = phi i1 [ %bar, %foo], [ %operand, %entry ]
  ret i1 %result
}

declare zeroext i1 @return_i1()

; CHECK-LABEL: call_test
define zeroext i1 @call_test() {
; CHECK: [[REG:%.+]] = call i1
  %result = call i1 @return_i1()
; CHECK: [[REG:%.+]] = zext i1 {{%.+}} to i64
; CHECK: [[REG:%.+]] = trunc i64 {{%.+}} to i1
; CHECK: ret i1 [[REG]]
  ret i1 %result
}
