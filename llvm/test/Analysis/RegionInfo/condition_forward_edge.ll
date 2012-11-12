; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @normal_condition() nounwind {
0:
	br label %"1"
1:
	br i1 1, label %"2", label %"3"
2:
	br label %"3"
3:
	ret void
}
; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK: [1] 1 => 3

; STAT: 2 region - The # of regions

; BBIT: 0, 1, 2, 3,
; BBIT: 1, 2,

; RNIT: 0, 1 => 3, 3,
; RNIT: 1, 2,
