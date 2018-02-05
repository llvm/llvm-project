; RUN: opt < %s -partial-inliner -S -skip-partial-inlining-cost-analysis | FileCheck %s
; RUN: opt < %s -passes=partial-inliner -S -skip-partial-inlining-cost-analysis | FileCheck %s

@stat = external global i32, align 4

define i32 @vararg(i32 %count, ...) {
entry:
  %vargs = alloca i8*, align 8
  %stat1 = load i32, i32* @stat, align 4
  %cmp = icmp slt i32 %stat1, 0
  br i1 %cmp, label %bb2, label %bb1

bb1:                                              ; preds = %entry
  %vg1 = add nsw i32 %stat1, 1
  store i32 %vg1, i32* @stat, align 4
  %vargs1 = bitcast i8** %vargs to i8*
  call void @llvm.va_start(i8* %vargs1)
  %va1 = va_arg i8** %vargs, i32
  call void @foo(i32 %count, i32 %va1) #2
  call void @llvm.va_end(i8* %vargs1)
  br label %bb2

bb2:                                              ; preds = %bb1, %entry
  %res = phi i32 [ 1, %bb1 ], [ 0, %entry ]
  ret i32 %res
}

declare void @foo(i32, i32)
declare void @llvm.va_start(i8*)
declare void @llvm.va_end(i8*)

define i32 @caller1(i32 %arg) {
bb:
  %tmp = tail call i32 (i32, ...) @vararg(i32 %arg)
  ret i32 %tmp
}
; CHECK-LABEL: @caller1
; CHECK: codeRepl.i:
; CHECK-NEXT:  call void (i32, i8**, i32, ...) @vararg.3_bb1(i32 %stat1.i, i8** %vargs.i, i32 %arg)

define i32 @caller2(i32 %arg, float %arg2) {
bb:
  %tmp = tail call i32 (i32, ...) @vararg(i32 %arg, i32 10, float %arg2)
  ret i32 %tmp
}

; CHECK-LABEL: @caller2
; CHECK: codeRepl.i:
; CHECK-NEXT:  call void (i32, i8**, i32, ...) @vararg.3_bb1(i32 %stat1.i, i8** %vargs.i, i32 %arg, i32 10, float %arg2)

; Test case to check that we do not extract a vararg function, if va_end is in
; a block that is not outlined.
define i32 @vararg_not_legal(i32 %count, ...) {
entry:
  %vargs = alloca i8*, align 8
  %vargs0 = bitcast i8** %vargs to i8*
  %stat1 = load i32, i32* @stat, align 4
  %cmp = icmp slt i32 %stat1, 0
  br i1 %cmp, label %bb2, label %bb1

bb1:                                              ; preds = %entry
  %vg1 = add nsw i32 %stat1, 1
  store i32 %vg1, i32* @stat, align 4
  %vargs1 = bitcast i8** %vargs to i8*
  call void @llvm.va_start(i8* %vargs1)
  %va1 = va_arg i8** %vargs, i32
  call void @foo(i32 %count, i32 %va1)
  br label %bb2

bb2:                                              ; preds = %bb1, %entry
  %res = phi i32 [ 1, %bb1 ], [ 0, %entry ]
  %ptr = phi i8* [ %vargs1, %bb1 ], [ %vargs0, %entry]
  call void @llvm.va_end(i8* %ptr)
  ret i32 %res
}

; CHECK-LABEL: @caller3
; CHECK: tail call i32 (i32, ...) @vararg_not_legal(i32 %arg, i32 %arg)
define i32 @caller3(i32 %arg) {
bb:
  %res = tail call i32 (i32, ...) @vararg_not_legal(i32 %arg, i32 %arg)
  ret i32 %res
}

declare i32* @err(i32*)

define signext i32 @vararg2(i32 * %l, ...) {
entry:
  br i1 undef, label %cleanup, label %cond.end

cond.end:                                         ; preds = %entry
  %call51 = call i32* @err(i32* nonnull %l)
  unreachable

cleanup:                                          ; preds = %entry
  ret i32 0
}

define i32* @caller_with_signext(i32* %foo) {
entry:
  %call1 = tail call signext i32 (i32*, ...) @vararg2(i32* %foo, i32 signext 8)
  unreachable
}

; CHECK-LABEL: @caller_with_signext
; CHECK: codeRepl.i:
; CHECK-NEXT:  call void (i32*, ...) @vararg2.1_cond.end(i32* %foo, i32 signext 8)
