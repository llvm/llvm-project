; RUN: opt -regions -analyze < %s | FileCheck %s

define void @normal_condition() nounwind {
0:
    br label %"7"
7:
    br i1 1, label %"1", label %"9"
9:
    br label %"8"
1:
    br i1 1, label %"6", label %"3"
6:
    br label %"9"
8:
    br label %"8"
3:
    br label %"4"
4:
    ret void
}

; CHECK:      Region tree:
; CHECK-NEXT: [0] 0 => <Function Return>
; CHECK-NEXT:   [1] 7 => 3
; CHECK-NEXT: End region tree
