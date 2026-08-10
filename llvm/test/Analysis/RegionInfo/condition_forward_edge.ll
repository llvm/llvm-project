; REQUIRES: asserts

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s
; RUN: opt < %s -passes='print<regions>' -stats 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -passes='print<regions>' -print-region-style=bb < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -passes='print<regions>' -print-region-style=rn < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @normal_condition() nounwind {
"0":
	br label %"1"
"1":
	br i1 1, label %"2", label %"3"
"2":
	br label %"3"
"3":
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
