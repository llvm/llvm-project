; REQUIRES: asserts
; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats -disable-output < %s 2>&1 | FileCheck -check-prefix=STAT %s
; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

define void @a_linear_impl_fig_1() nounwind {
0:
        br label %"1"
1:
	br label %"2"
2:
	br label %"3"
3:
	br i1 1, label %"13", label %"4"
4:
	br i1 1, label %"5", label %"1"
5:
	br i1 1, label %"8", label %"6"
6:
	br i1 1, label %"7", label %"4"
7:
	ret void
8:
	br i1 1, label %"9", label %"1"
9:
	br label %"10"
10:
	br i1 1, label %"12", label %"11"
11:
	br i1 1, label %"9", label %"8"
13:
	br i1 1, label %"2", label %"1"
12:
 	switch i32 0, label %"1" [ i32 0, label %"9"
                                  i32 1, label %"8"]
}

; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK-NEXT: [1] 1 => 7
; CHECK-NEXT:   [2] 1 => 4
; CHECK-NEXT:   [2] 8 => 1

; STAT: 4 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: 0, 1, 2, 3, 13, 4, 5, 8, 9, 10, 12, 11, 6, 7,
; BBIT: 1, 2, 3, 13, 4, 5, 8, 9, 10, 12, 11, 6,
; BBIT: 1, 2, 3, 13,
; BBIT: 8, 9, 10, 12, 11,

; RNIT: 0, 1 => 7, 7,
; RNIT: 1 => 4, 4, 5, 8 => 1, 6,
; RNIT: 1, 2, 3, 13,
; RNIT: 8, 9, 10, 12, 11,
