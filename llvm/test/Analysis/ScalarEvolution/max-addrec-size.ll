; RUN: opt -disable-output "-passes=print<scalar-evolution>" -scalar-evolution-max-add-rec-size=3 < %s 2>&1 | FileCheck %s

; Show that we are able to avoid creation of huge SCEVs by capping the max
; AddRec size.
define i32 @test_01(i32 %a, i32 %b) {

; CHECK-LABEL: Classifying expressions for: @test_01
; CHECK-NEXT:    %iv = phi i32 [ %a, %entry ], [ %iv.next, %loop ]
; CHECK-NEXT:    -->  {%a,+,%b}<%loop> U: full-set S: full-set
; CHECK-NEXT:    %iv.next = add i32 %iv, %b
; CHECK-NEXT:    -->  {(%a + %b),+,%b}<%loop> U: full-set S: full-set
; CHECK-NEXT:    %x1 = mul i32 %iv, %iv.next
; CHECK-NEXT:    -->  {((%a + %b) * %a),+,(((2 * %a) + (2 * %b)) * %b),+,(2 * %b * %b)}<%loop> U: full-set S: full-set
; CHECK-NEXT:    %x2 = mul i32 %x1, %x1
; CHECK-NEXT:    -->  ({((%a + %b) * %a),+,(((2 * %a) + (2 * %b)) * %b),+,(2 * %b * %b)}<%loop> * {((%a + %b) * %a),+,(((2 * %a) + (2 * %b)) * %b),+,(2 * %b * %b)}<%loop>) U: full-set S: full-set
; CHECK-NEXT:    %x3 = mul i32 %x2, %x1
; CHECK-NEXT:    -->  ({((%a + %b) * %a),+,(((2 * %a) + (2 * %b)) * %b),+,(2 * %b * %b)}<%loop> * {((%a + %b) * %a),+,(((2 * %a) + (2 * %b)) * %b),+,(2 * %b * %b)}<%loop> * {((%a + %b) * %a),+,(((2 * %a) + (2 * %b)) * %b),+,(2 * %b * %b)}<%loop>) U: full-set S: full-set

entry:
  br label %loop

loop:
  %iv = phi i32 [ %a, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, %b
  %cond = icmp slt i32 %iv.next, 1000
  br i1 %cond, label %loop, label %exit

exit:
  %x1 = mul i32 %iv, %iv.next
  %x2 = mul i32 %x1, %x1
  %x3 = mul i32 %x2, %x1
  ret i32 %x3
}
