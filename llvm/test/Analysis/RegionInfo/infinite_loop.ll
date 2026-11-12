; REQUIRES: asserts
; RUN: opt -stats -passes='print<regions>' -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s

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
; STAT: 1 region - The # of regions
