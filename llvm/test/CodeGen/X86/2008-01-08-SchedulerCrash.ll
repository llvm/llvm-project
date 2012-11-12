; RUN: llc < %s -march=x86 -mattr=+cmov | FileCheck %s
;
; Test scheduling a multi-use compare. We should neither spill flags
; nor clone the compare.
; CHECK: cmp
; CHECK-NOT: pushf
; CHECK: cmov
; CHECK-NOT: cmp
; CHECK: cmov

	%struct.indexentry = type { i32, i8*, i8*, i8*, i8*, i8* }

define i32 @_bfd_stab_section_find_nearest_line(i32 %offset, i1 %cond) nounwind  {
entry:
	%tmp910 = add i32 0, %offset		; <i32> [#uses=1]
	br i1 %cond, label %bb951, label %bb917

bb917:		; preds = %entry
	ret i32 0

bb951:		; preds = %bb986, %entry
	%tmp955 = sdiv i32 0, 2		; <i32> [#uses=3]
	%tmp961 = getelementptr %struct.indexentry* null, i32 %tmp955, i32 0		; <i32*> [#uses=1]
	br i1 %cond, label %bb986, label %bb967

bb967:		; preds = %bb951
	ret i32 0

bb986:		; preds = %bb951
	%tmp993 = load i32* %tmp961, align 4		; <i32> [#uses=1]
	%tmp995 = icmp ugt i32 %tmp993, %tmp910		; <i1> [#uses=2]
	%tmp1002 = add i32 %tmp955, 1		; <i32> [#uses=1]
	%low.0 = select i1 %tmp995, i32 0, i32 %tmp1002		; <i32> [#uses=1]
	%high.0 = select i1 %tmp995, i32 %tmp955, i32 0		; <i32> [#uses=1]
	%tmp1006 = icmp eq i32 %low.0, %high.0		; <i1> [#uses=1]
	br i1 %tmp1006, label %UnifiedReturnBlock, label %bb951

UnifiedReturnBlock:		; preds = %bb986
	ret i32 1
}
