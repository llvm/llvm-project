; RUN: llc < %s -mtriple=i686--
; PR1975

@nodes = external global i64		; <i64*> [#uses=2]

define fastcc i32 @ab(i32 %alpha, i32 %beta) nounwind  {
entry:
	%tmp1 = load i64, i64* @nodes, align 8		; <i64> [#uses=1]
	%tmp2 = add i64 %tmp1, 1		; <i64> [#uses=1]
	store i64 %tmp2, i64* @nodes, align 8
	ret i32 0
}
