; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s

define void @merge_values_with_ranges(i32 *%a_len_ptr, i32 *%b_len_ptr, i1 %unknown_cond) {

; CHECK-LABEL: Classifying expressions for: @merge_values_with_ranges
; CHECK:       %len = phi i32 [ %len_a, %if.true ], [ %len_b, %if.false ]
; CHECK-NEXT:  -->  %len U: [0,2147483647) S: [0,2147483647)

 entry:
  br i1 %unknown_cond, label %if.true, label %if.false

if.true:
  %len_a = load i32, i32* %a_len_ptr, !range !0
  br label %merge

if.false:
  %len_b = load i32, i32* %b_len_ptr, !range !0
  br label %merge

merge:
  %len = phi i32 [ %len_a, %if.true ], [ %len_b, %if.false ]
  ret void
}

define void @merge_values_with_ranges_looped(i32 *%a_len_ptr, i32 *%b_len_ptr) {

; TODO: We could be much smarter here. So far we just make sure that we do not
;       go into infinite loop analyzing these Phis.

; CHECK-LABEL: Classifying expressions for: @merge_values_with_ranges_looped
; CHECK:       %p1 = phi i32 [ %len_a, %entry ], [ %p2, %loop ]
; CHECK-NEXT:  -->  %p1 U: full-set S: full-set
; CHECK:         %p2 = phi i32 [ %len_b, %entry ], [ %p1, %loop ]
; CHECK-NEXT:  -->  %p2 U: full-set S: full-set

 entry:
  %len_a = load i32, i32* %a_len_ptr, !range !0
  %len_b = load i32, i32* %b_len_ptr, !range !0
  br label %loop

loop:
  %p1 = phi i32 [ %len_a, %entry ], [ %p2, %loop ]
  %p2 = phi i32 [ %len_b, %entry ], [ %p1, %loop ]
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  %loop.cond = icmp slt i32 %iv.next, 100
  br i1 %loop.cond, label %loop, label %exit

exit:
  ret void
}


!0 = !{i32 0, i32 2147483647}
