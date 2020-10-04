; REQUIRES: asserts
; RUN: opt -regions -analyze -enable-new-pm=0 < %s 
; RUN: opt -passes='print<regions>' -disable-output < %s 2>&1
; RUN: opt -regions -stats -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s

define void @normal_condition() nounwind {
0:
	br label %1
1:
	br i1 1, label %2, label %3
2:
	br label %2
3:
	br label %4
4:
	ret void
}
; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; STAT: 1 region - The # of regions
