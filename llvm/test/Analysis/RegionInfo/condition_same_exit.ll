; REQUIRES: asserts
; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

define void @normal_condition() nounwind {
0:
	br i1 1, label %"1", label %"4"

1:
	br i1 1, label %"2", label %"3"
2:
	br label %"4"
3:
	br label %"4"
4:
	ret void
}
; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK-NEXT: [1] 0 => 4
; CHECK-NEXT:   [2] 1 => 4
; STAT: 3 region - The # of regions

; BBIT: 0, 1, 2, 4, 3,
; BBIT: 0, 1, 2, 3,
; BBIT: 1, 2, 3,

; RNIT: 0 => 4, 4,
; RNIT: 0, 1 => 4,
; RNIT: 1, 2, 3,
