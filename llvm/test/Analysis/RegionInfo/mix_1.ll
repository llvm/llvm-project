; REQUIRES: asserts
; RUN: opt -regions -analyze < %s | FileCheck %s
; RUN: opt -regions -stats < %s 2>&1 | FileCheck -check-prefix=STAT %s

; RUN: opt -regions -print-region-style=bb  -analyze < %s 2>&1 | FileCheck -check-prefix=BBIT %s
; RUN: opt -regions -print-region-style=rn  -analyze < %s 2>&1 | FileCheck -check-prefix=RNIT %s

; RUN: opt < %s -passes='print<regions>' 2>&1 | FileCheck %s

define void @a_linear_impl_fig_1() nounwind {
0:

	br i1 1, label %"1", label %"15"
1:
 	switch i32 0, label %"2" [ i32 0, label %"3"
                                  i32 1, label %"7"]
2:
	br label %"4"
3:
	br label %"5"
4:
	br label %"6"
5:
	br label %"6"
6:
	br label %"7"
7:
	br label %"15"
15:
	br label %"8"
8:
	br label %"16"
16:
        br label %"9"
9:
	br i1 1, label %"10", label %"11"
11:
	br i1 1, label %"13", label %"12"
13:
	br label %"14"
12:
	br label %"14"
14:
	br label %"8"
10:
        br label %"17"
17:
        br label %"18"
18:
        ret void
}

; CHECK-NOT: =>
; CHECK: [0] 0 => <Function Return>
; CHECK-NEXT: [1] 0 => 15
; CHECK-NEXT:  [2] 1 => 7
; CHECK-NEXT: [1] 8 => 10
; CHECK-NEXT:  [2] 11 => 14
; STAT: 5 region - The # of regions
; STAT: 1 region - The # of simple regions

; BBIT: 0, 1, 2, 4, 6, 7, 15, 8, 16, 9, 10, 17, 18, 11, 13, 14, 12, 3, 5,
; BBIT: 0, 1, 2, 4, 6, 7, 3, 5,
; BBIT: 1, 2, 4, 6, 3, 5,
; BBIT: 8, 16, 9, 11, 13, 14, 12,
; BBIT: 11, 13, 12,

; RNIT: 0 => 15, 15, 8 => 10, 10, 17, 18,
; RNIT: 0, 1 => 7, 7,
; RNIT: 1, 2, 4, 6, 3, 5,
; RNIT: 8, 16, 9, 11 => 14, 14,
; RNIT: 11, 13, 12,
