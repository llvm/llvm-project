; RUN: llc < %s
; REQUIRES: default_triple

; Examples that exhibits a bug in DAGCombine.  The case is triggered by the
; following program.  The bug is DAGCombine assumes that the bit convert
; preserves the number of elements so the optimization code tries to read
; through the 3rd mask element, which doesn't exist.
define i32 @update(<2 x i64> %val1, <2 x i64> %val2) nounwind readnone {
entry:
	%shuf = shufflevector <2 x i64> %val1, <2 x i64> %val2, <2 x i32> <i32 0, i32 3>
	%bit  = bitcast <2 x i64> %shuf to <4 x i32>
	%res =  extractelement <4 x i32> %bit, i32 3
	ret i32 %res
}
