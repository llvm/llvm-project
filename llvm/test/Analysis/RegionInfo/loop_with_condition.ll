; REQUIRES: asserts

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @normal_condition() nounwind {
"0":
        br label %"1"
"1":
	br i1 1, label %"6", label %"2"
"2":
	br i1 1, label %"3", label %"4"
"3":
	br label %"5"
"4":
	br label %"5"
"5":
        br label %"8"
"8":
        br i1 1, label %"7", label %"9"
"9":
        br label %"2"
"7":
        br label %"6"
"6":
	ret void
}

; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK-NEXT: [1] 1 => 6
; CHECK-NEXT:   [2] 2 => 7
; CHECK-NEXT:     [3] 2 => 5
; STAT: 4 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: 0, 1, 6, 2, 3, 5, 8, 7, 9, 4,
; BBIT: 1, 2, 3, 5, 8, 7, 9, 4,
; BBIT: 2, 3, 5, 8, 9, 4,
; BBIT: 2, 3, 4,

; RNIT: 0, 1 => 6, 6,
; RNIT: 1, 2 => 7, 7,
; RNIT: 2 => 5, 5, 8, 9,
; RNIT: 2, 3, 4,
