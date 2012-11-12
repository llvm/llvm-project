; RUN: llc < %s -mtriple=x86_64-apple-darwin | grep call | grep 560
; rdar://6522427

	%"struct.clang::Action" = type { %"struct.clang::ActionBase" }
	%"struct.clang::ActionBase" = type { i32 (...)** }
	%"struct.clang::ActionBase::ActionResult<0u>" = type { i8*, i8 }
@NumTrials = internal global i32 10000000		; <i32*> [#uses=2]
@llvm.used = appending global [1 x i8*] [ i8* bitcast (void (i8*, %"struct.clang::Action"*)* @_Z25RawPointerPerformanceTestPvRN5clang6ActionE to i8*) ], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define void @_Z25RawPointerPerformanceTestPvRN5clang6ActionE(i8* %Val, %"struct.clang::Action"* %Actions) nounwind {
entry:
	%0 = alloca %"struct.clang::ActionBase::ActionResult<0u>", align 8		; <%"struct.clang::ActionBase::ActionResult<0u>"*> [#uses=3]
	%1 = load i32* @NumTrials, align 4		; <i32> [#uses=1]
	%2 = icmp eq i32 %1, 0		; <i1> [#uses=1]
	br i1 %2, label %return, label %bb.nph

bb.nph:		; preds = %entry
	%3 = getelementptr %"struct.clang::Action"* %Actions, i64 0, i32 0, i32 0		; <i32 (...)***> [#uses=1]
	%mrv_gep = bitcast %"struct.clang::ActionBase::ActionResult<0u>"* %0 to i64*		; <i64*> [#uses=1]
	%mrv_gep1 = getelementptr %"struct.clang::ActionBase::ActionResult<0u>"* %0, i64 0, i32 1		; <i8*> [#uses=1]
	%4 = bitcast i8* %mrv_gep1 to i64*		; <i64*> [#uses=1]
	%5 = getelementptr %"struct.clang::ActionBase::ActionResult<0u>"* %0, i64 0, i32 0		; <i8**> [#uses=1]
	br label %bb

bb:		; preds = %bb, %bb.nph
	%Trial.01 = phi i32 [ 0, %bb.nph ], [ %12, %bb ]		; <i32> [#uses=1]
	%Val_addr.02 = phi i8* [ %Val, %bb.nph ], [ %11, %bb ]		; <i8*> [#uses=1]
	%6 = load i32 (...)*** %3, align 8		; <i32 (...)**> [#uses=1]
	%7 = getelementptr i32 (...)** %6, i64 70		; <i32 (...)**> [#uses=1]
	%8 = load i32 (...)** %7, align 8		; <i32 (...)*> [#uses=1]
	%9 = bitcast i32 (...)* %8 to { i64, i64 } (%"struct.clang::Action"*, i8*)*		; <{ i64, i64 } (%"struct.clang::Action"*, i8*)*> [#uses=1]
	%10 = call { i64, i64 } %9(%"struct.clang::Action"* %Actions, i8* %Val_addr.02) nounwind		; <{ i64, i64 }> [#uses=2]
	%mrv_gr = extractvalue { i64, i64 } %10, 0		; <i64> [#uses=1]
	store i64 %mrv_gr, i64* %mrv_gep
	%mrv_gr2 = extractvalue { i64, i64 } %10, 1		; <i64> [#uses=1]
	store i64 %mrv_gr2, i64* %4
	%11 = load i8** %5, align 8		; <i8*> [#uses=1]
	%12 = add i32 %Trial.01, 1		; <i32> [#uses=2]
	%13 = load i32* @NumTrials, align 4		; <i32> [#uses=1]
	%14 = icmp ult i32 %12, %13		; <i1> [#uses=1]
	br i1 %14, label %bb, label %return

return:		; preds = %bb, %entry
	ret void
}
