; RUN: llvm-as < %s | llc -march=x86 -mattr=+sse2 -pre-alloc-split -stats |& \
; RUN:   grep {pre-alloc-split} | grep {Number of intervals split} | grep 1

define i32 @t(i32 %arg) {
entry:
	br label %bb6

.noexc6:		; preds = %bb6
	%0 = and i32 %2, -8		; <i32> [#uses=1]
	tail call void @llvm.memmove.i32(i8* %3, i8* null, i32 %0, i32 1) nounwind
	store double %1, double* null, align 8
	br label %bb6

bb6:		; preds = %.noexc6, %entry
	%1 = uitofp i32 %arg to double		; <double> [#uses=1]
	%2 = sub i32 0, 0		; <i32> [#uses=1]
	%3 = invoke i8* @_Znwm(i32 0)
			to label %.noexc6 unwind label %lpad32		; <i8*> [#uses=1]

lpad32:		; preds = %bb6
	unreachable
}

declare void @llvm.memmove.i32(i8*, i8*, i32, i32) nounwind

declare i8* @_Znwm(i32)
