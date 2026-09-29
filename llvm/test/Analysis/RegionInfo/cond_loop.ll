; REQUIRES: asserts

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @normal_condition() nounwind {
"5":
        br label %"0"

"0":
	br label %"1"
"1":
	br i1 1, label %"2", label %"3"
"2":
	ret void
"3":
	br i1 1, label %"1", label %"4"
"4":
        br label %"0"
}

; CHECK-NOT: =>
; CHECK: [0] 5 => <Function Return>
; CHECK: [1] 0 => 2

; STAT: 2 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: 5, 0, 1, 2, 3, 4,
; BBIT: 0, 1, 3, 4,

; RNIT: 5, 0 => 2, 2,
; RNIT: 0, 1, 3, 4,
