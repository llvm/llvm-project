; REQUIRES: asserts
; RUN: opt -regions -analyze -enable-new-pm=0 < %s 
; RUN: opt -regions -stats -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze -enable-new-pm=0 < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze -enable-new-pm=0 < %s 2>&1 | FileCheck -check-prefix=RNIT %s
; RUN: opt -passes='print<regions>' -disable-output < %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @normal_condition() nounwind {
"0":
	br label %"1"
"1":
	br i1 1, label %"2", label %"3"
"2":
	br label %"5"
"5":
	br i1 1, label %"11", label %"12"
"11":
        br label %"6"
"12":
        br label %"6"
"6":
        br label %"2"
"3":
	br label %"4"
"4":
	ret void
}
; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK-NOT: [1]
; STAT: 1 region - The # of regions

; BBIT:  0, 1, 2, 5, 11, 6, 12, 3, 4,

; RNIT: 0, 1, 2, 5, 11, 6, 12, 3, 4,
