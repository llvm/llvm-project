; RUN: opt -regions -analyze < %s 
; RUN: opt -regions -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s

; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

define void @normal_condition() nounwind {
0:
	br label %"7"
7:
	br i1 1, label %"1", label %"8"
1:
	br i1 1, label %"2", label %"3"
2:
	br label %"5"
5:
	br i1 1, label %"11", label %"12"
11:
        br label %"6"
12:
        br label %"6"
6:
        br label %"2"
8:
	br label %"9"
9:
	br i1 1, label %"13", label %"14"
13:
        br label %"10"
14:
        br label %"10"
10:
        br label %"8"
3:
	br label %"4"
4:
	ret void
}
; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK-NEXT: [1] 1 => 3
; CHECK-NEXT: [1] 7 => 1
; STAT: 3 region - The # of regions
; STAT: 2 region - The # of simple regions

; BBIT: 0, 7, 1, 2, 5, 11, 6, 12, 3, 4, 8, 9, 13, 10, 14,
; BBIT: 7, 8, 9, 13, 10, 14,
; BBIT: 1, 2, 5, 11, 6, 12,

; RNIT: 0, 7 => 1, 1 => 3, 3, 4,
; RNIT: 7, 8, 9, 13, 10, 14,
; RNIT: 1, 2, 5, 11, 6, 12,
