; REQUIRES: asserts

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @normal_condition() nounwind {
"0":
	br label %"7"
"7":
	br i1 1, label %"1", label %"8"
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
	br i1 1, label %"2", label %"10"
"8":
	br label %"9"
"9":
	br i1 1, label %"13", label %"14"
"13":
        br label %"10"
"14":
        br label %"10"
"10":
        br label %"8"
"3":
	br label %"4"
"4":
	ret void
}
; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK-NEXT: [1] 2 => 10
; CHECK-NEXT: [2] 5 => 6
; STAT: 3 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: 0, 7, 1, 2, 5, 11, 6, 10, 8, 9, 13, 14, 12, 3, 4,
; BBIT: 2, 5, 11, 6, 12,
; BBIT: 5, 11, 12,
; RNIT: 0, 7, 1, 2 => 10, 10, 8, 9, 13, 14, 3, 4,
; RNIT: 2, 5 => 6, 6,
; RNIT: 5, 11, 12,
