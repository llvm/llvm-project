; RUN: llc < %s
;; Formerly crashed, see PR 1508
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128"
target triple = "powerpc64-apple-darwin8"
	%struct.Range = type { i64, i64 }

define void @Bork(i64 %range.0.0, i64 %range.0.1, i64 %size) {
entry:
	%effectiveRange = alloca %struct.Range, align 8		; <%struct.Range*> [#uses=2]
	%tmp4 = call i8* @llvm.stacksave()		; <i8*> [#uses=1]
	%size1 = trunc i64 %size to i32		; <i32> [#uses=1]
	%tmp17 = alloca i8*, i32 %size1		; <i8**> [#uses=1]
	invoke void @Foo(i8** %tmp17)
			to label %bb30.preheader unwind label %unwind

bb30.preheader:		; preds = %entry
	%tmp26 = getelementptr %struct.Range* %effectiveRange, i64 0, i32 1		; <i64*> [#uses=1]
	br label %bb30

unwind:		; preds = %cond_true, %entry
        %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
                 catch i8* null
	call void @llvm.stackrestore(i8* %tmp4)
        resume { i8*, i32 } %exn

invcont23:		; preds = %cond_true
	%tmp27 = load i64* %tmp26, align 8		; <i64> [#uses=1]
	%tmp28 = sub i64 %range_addr.1.0, %tmp27		; <i64> [#uses=1]
	br label %bb30

bb30:		; preds = %invcont23, %bb30.preheader
	%range_addr.1.0 = phi i64 [ %tmp28, %invcont23 ], [ %range.0.1, %bb30.preheader ]		; <i64> [#uses=2]
	%tmp33 = icmp eq i64 %range_addr.1.0, 0		; <i1> [#uses=1]
	br i1 %tmp33, label %cleanup, label %cond_true

cond_true:		; preds = %bb30
	invoke void @Bar(i64 %range.0.0, %struct.Range* %effectiveRange)
			to label %invcont23 unwind label %unwind

cleanup:		; preds = %bb30
	ret void
}

declare i8* @llvm.stacksave() nounwind

declare void @Foo(i8**)

declare void @Bar(i64, %struct.Range*)

declare void @llvm.stackrestore(i8*) nounwind

declare i32 @__gxx_personality_v0(...)
