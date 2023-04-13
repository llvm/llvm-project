; RUN: opt -disable-output "-passes=print<scalar-evolution>" < %s 2>&1 | FileCheck %s

; This test set ensures that we can correctly operate with recurrencies from
; different loops.

; Check that we can evaluate a sum of phis from two different loops in any
; order.

define void @test_00() {

; CHECK-LABEL: Classifying expressions for: @test_00
; CHECK:       %sum1 = add i32 %phi1, %phi2
; CHECK-NEXT:  -->  {14,+,3}<%loop1>
; CHECK:       %sum2 = add i32 %sum1, %phi3
; CHECK-NEXT:  -->  {20,+,6}<%loop1>
; CHECK:       %sum3 = add i32 %phi4, %phi5
; CHECK-NEXT:  -->  {116,+,3}<%loop2>
; CHECK:       %sum4 = add i32 %sum3, %phi6
; CHECK-NEXT:  -->  {159,+,6}<%loop2>
; CHECK:       %s1 = add i32 %phi1, %phi4
; CHECK-NEXT:  -->  {{{{}}73,+,1}<nuw><nsw><%loop1>,+,1}<nw><%loop2>
; CHECK:       %s2 = add i32 %phi5, %phi2
; CHECK-NEXT:  -->  {{{{}}57,+,2}<nuw><nsw><%loop1>,+,2}<nw><%loop2>
; CHECK:       %s3 = add i32 %sum1, %sum3
; CHECK-NEXT:  -->  {{{{}}130,+,3}<%loop1>,+,3}<%loop2>
; CHECK:       %s4 = add i32 %sum4, %sum2
; CHECK-NEXT:  -->  {{{{}}179,+,6}<%loop1>,+,6}<%loop2>
; CHECK:       %s5 = add i32 %phi3, %sum3
; CHECK-NEXT:  -->  {{{{}}122,+,3}<nuw><nsw><%loop1>,+,3}<%loop2>
; CHECK:       %s6 = add i32 %sum2, %phi6
; CHECK-NEXT:  -->  {{{{}}63,+,6}<%loop1>,+,3}<nw><%loop2>

entry:
  br label %loop1

loop1:
  %phi1 = phi i32 [ 10, %entry ], [ %phi1.inc, %loop1 ]
  %phi2 = phi i32 [ 4, %entry ], [ %phi2.inc, %loop1 ]
  %phi3 = phi i32 [ 6, %entry ], [ %phi3.inc, %loop1 ]
  %phi1.inc = add i32 %phi1, 1
  %phi2.inc = add i32 %phi2, 2
  %phi3.inc = add i32 %phi3, 3
  %sum1 = add i32 %phi1, %phi2
  %sum2 = add i32 %sum1, %phi3
  %cond1 = icmp ult i32 %sum2, 1000
  br i1 %cond1, label %loop1, label %loop2

loop2:
  %phi4 = phi i32 [ 63, %loop1 ], [ %phi4.inc, %loop2 ]
  %phi5 = phi i32 [ 53, %loop1 ], [ %phi5.inc, %loop2 ]
  %phi6 = phi i32 [ 43, %loop1 ], [ %phi6.inc, %loop2 ]
  %phi4.inc = add i32 %phi4, 1
  %phi5.inc = add i32 %phi5, 2
  %phi6.inc = add i32 %phi6, 3
  %sum3 = add i32 %phi4, %phi5
  %sum4 = add i32 %sum3, %phi6
  %cond2 = icmp ult i32 %sum4, 1000
  br i1 %cond2, label %loop2, label %exit

exit:
  %s1 = add i32 %phi1, %phi4
  %s2 = add i32 %phi5, %phi2
  %s3 = add i32 %sum1, %sum3
  %s4 = add i32 %sum4, %sum2
  %s5 = add i32 %phi3, %sum3
  %s6 = add i32 %sum2, %phi6
  ret void
}

; Check that we can evaluate a sum of phis+invariants from two different loops
; in any order.

define void @test_01(i32 %a, i32 %b) {

; CHECK-LABEL: Classifying expressions for: @test_01
; CHECK:       %sum1 = add i32 %phi1, %phi2
; CHECK-NEXT:  -->  {(%a + %b),+,3}<%loop1>
; CHECK:       %sum2 = add i32 %sum1, %phi3
; CHECK-NEXT:  -->  {(6 + %a + %b),+,6}<%loop1>
; CHECK:       %is1 = add i32 %sum2, %a
; CHECK-NEXT:  -->  {(6 + (2 * %a) + %b),+,6}<%loop1>
; CHECK:       %sum3 = add i32 %phi4, %phi5
; CHECK-NEXT:  -->  {116,+,3}<%loop2>
; CHECK:       %sum4 = add i32 %sum3, %phi6
; CHECK-NEXT:  -->  {159,+,6}<%loop2>
; CHECK:       %is2 = add i32 %sum4, %b
; CHECK-NEXT:  -->  {(159 + %b),+,6}<%loop2>
; CHECK:       %ec2 = add i32 %is1, %is2
; CHECK-NEXT:  -->  {{{{}}(165 + (2 * %a) + (2 * %b)),+,6}<%loop1>,+,6}<%loop2>
; CHECK:       %s1 = add i32 %phi1, %is1
; CHECK-NEXT:  -->  {(6 + (3 * %a) + %b),+,7}<%loop1>
; CHECK:       %s2 = add i32 %is2, %phi4
; CHECK-NEXT:  -->  {(222 + %b),+,7}<%loop2>
; CHECK:       %s3 = add i32 %is1, %phi5
; CHECK-NEXT:  -->  {{{{}}(59 + (2 * %a) + %b),+,6}<%loop1>,+,2}<nw><%loop2>
; CHECK:       %s4 = add i32 %phi2, %is2
; CHECK-NEXT:  -->  {{{{}}(159 + (2 * %b)),+,2}<nw><%loop1>,+,6}<%loop2>
; CHECK:       %s5 = add i32 %is1, %is2
; CHECK-NEXT:  -->  {{{{}}(165 + (2 * %a) + (2 * %b)),+,6}<%loop1>,+,6}<%loop2>
; CHECK:       %s6 = add i32 %is2, %is1
; CHECK-NEXT:  -->  {{{{}}(165 + (2 * %a) + (2 * %b)),+,6}<%loop1>,+,6}<%loop2>

entry:
  br label %loop1

loop1:
  %phi1 = phi i32 [ %a, %entry ], [ %phi1.inc, %loop1 ]
  %phi2 = phi i32 [ %b, %entry ], [ %phi2.inc, %loop1 ]
  %phi3 = phi i32 [ 6, %entry ], [ %phi3.inc, %loop1 ]
  %phi1.inc = add i32 %phi1, 1
  %phi2.inc = add i32 %phi2, 2
  %phi3.inc = add i32 %phi3, 3
  %sum1 = add i32 %phi1, %phi2
  %sum2 = add i32 %sum1, %phi3
  %is1 = add i32 %sum2, %a
  %cond1 = icmp ult i32 %is1, 1000
  br i1 %cond1, label %loop1, label %loop2

loop2:
  %phi4 = phi i32 [ 63, %loop1 ], [ %phi4.inc, %loop2 ]
  %phi5 = phi i32 [ 53, %loop1 ], [ %phi5.inc, %loop2 ]
  %phi6 = phi i32 [ 43, %loop1 ], [ %phi6.inc, %loop2 ]
  %phi4.inc = add i32 %phi4, 1
  %phi5.inc = add i32 %phi5, 2
  %phi6.inc = add i32 %phi6, 3
  %sum3 = add i32 %phi4, %phi5
  %sum4 = add i32 %sum3, %phi6
  %is2 = add i32 %sum4, %b
  %ec2 = add i32 %is1, %is2
  %cond2 = icmp ult i32 %ec2, 1000
  br i1 %cond2, label %loop2, label %exit

exit:
  %s1 = add i32 %phi1, %is1
  %s2 = add i32 %is2, %phi4
  %s3 = add i32 %is1, %phi5
  %s4 = add i32 %phi2, %is2
  %s5 = add i32 %is1, %is2
  %s6 = add i32 %is2, %is1
  ret void
}

; Check that we can correctly evaluate a sum of phis+variants from two different
; loops in any order.

define void @test_02(i32 %a, i32 %b, ptr %p) {

; CHECK-LABEL: Classifying expressions for: @test_02
; CHECK:       %sum1 = add i32 %phi1, %phi2
; CHECK-NEXT:  -->  {(%a + %b),+,3}<%loop1>
; CHECK:       %sum2 = add i32 %sum1, %phi3
; CHECK-NEXT:  -->  {(6 + %a + %b),+,6}<%loop1>
; CHECK:       %is1 = add i32 %sum2, %v1
; CHECK-NEXT:  -->  ({(6 + %a + %b),+,6}<%loop1> + %v1)
; CHECK:       %sum3 = add i32 %phi4, %phi5
; CHECK-NEXT:  -->  {(%a + %b),+,3}<%loop2>
; CHECK:       %sum4 = add i32 %sum3, %phi6
; CHECK-NEXT:  -->  {(43 + %a + %b),+,6}<%loop2>
; CHECK:       %is2 = add i32 %sum4, %v2
; CHECK-NEXT:  -->  ({(43 + %a + %b),+,6}<%loop2> + %v2)
; CHECK:       %is3 = add i32 %v1, %sum2
; CHECK-NEXT:  -->  ({(6 + %a + %b),+,6}<%loop1> + %v1)
; CHECK:       %ec2 = add i32 %is1, %is3
; CHECK-NEXT:  -->  (2 * ({(6 + %a + %b),+,6}<%loop1> + %v1))
; CHECK:       %s1 = add i32 %phi1, %is1
; CHECK-NEXT:  -->  ({(6 + (2 * %a) + %b),+,7}<%loop1> + %v1)
; CHECK:       %s2 = add i32 %is2, %phi4
; CHECK-NEXT:  -->  ({(43 + (2 * %a) + %b),+,7}<%loop2> + %v2)
; CHECK:       %s3 = add i32 %is1, %phi5
; CHECK-NEXT:  -->  {({(6 + (2 * %b) + %a),+,6}<%loop1> + %v1),+,2}<%loop2>
; CHECK:       %s4 = add i32 %phi2, %is2
; CHECK-NEXT:  -->  ({{{{}}(43 + (2 * %b) + %a),+,2}<%loop1>,+,6}<%loop2> + %v2)
; CHECK:       %s5 = add i32 %is1, %is2
; CHECK-NEXT:  -->  ({({(49 + (2 * %a) + (2 * %b)),+,6}<%loop1> + %v1),+,6}<%loop2> + %v2)
; CHECK:       %s6 = add i32 %is2, %is1
; CHECK-NEXT:  -->  ({({(49 + (2 * %a) + (2 * %b)),+,6}<%loop1> + %v1),+,6}<%loop2> + %v2)

entry:
  br label %loop1

loop1:
  %phi1 = phi i32 [ %a, %entry ], [ %phi1.inc, %loop1 ]
  %phi2 = phi i32 [ %b, %entry ], [ %phi2.inc, %loop1 ]
  %phi3 = phi i32 [ 6, %entry ], [ %phi3.inc, %loop1 ]
  %phi1.inc = add i32 %phi1, 1
  %phi2.inc = add i32 %phi2, 2
  %phi3.inc = add i32 %phi3, 3
  %v1 = load i32, ptr %p
  %sum1 = add i32 %phi1, %phi2
  %sum2 = add i32 %sum1, %phi3
  %is1 = add i32 %sum2, %v1
  %cond1 = icmp ult i32 %is1, 1000
  br i1 %cond1, label %loop1, label %loop2

loop2:
  %phi4 = phi i32 [ %a, %loop1 ], [ %phi4.inc, %loop2 ]
  %phi5 = phi i32 [ %b, %loop1 ], [ %phi5.inc, %loop2 ]
  %phi6 = phi i32 [ 43, %loop1 ], [ %phi6.inc, %loop2 ]
  %phi4.inc = add i32 %phi4, 1
  %phi5.inc = add i32 %phi5, 2
  %phi6.inc = add i32 %phi6, 3
  %v2 = load i32, ptr %p
  %sum3 = add i32 %phi4, %phi5
  %sum4 = add i32 %sum3, %phi6
  %is2 = add i32 %sum4, %v2
  %is3 = add i32 %v1, %sum2
  %ec2 = add i32 %is1, %is3
  %cond2 = icmp ult i32 %ec2, 1000
  br i1 %cond2, label %loop2, label %exit

exit:
  %s1 = add i32 %phi1, %is1
  %s2 = add i32 %is2, %phi4
  %s3 = add i32 %is1, %phi5
  %s4 = add i32 %phi2, %is2
  %s5 = add i32 %is1, %is2
  %s6 = add i32 %is2, %is1
  ret void
}

; Mix of previous use cases that demonstrates %s3 can be incorrectly treated as
; a recurrence of loop1 because of operands order if we pick recurrencies in an
; incorrect order. It also shows that we cannot safely fold v1 (SCEVUnknown)
; because we cannot prove for sure that it doesn't use Phis of loop 2.

define void @test_03(i32 %a, i32 %b, i32 %c, ptr %p) {

; CHECK-LABEL: Classifying expressions for: @test_03
; CHECK:       %v1 = load i32, ptr %p
; CHECK-NEXT:  -->  %v1
; CHECK:       %s1 = add i32 %phi1, %v1
; CHECK-NEXT:  -->  ({%a,+,1}<%loop1> + %v1)
; CHECK:       %s2 = add i32 %s1, %b
; CHECK-NEXT:  -->  ({(%a + %b),+,1}<%loop1> + %v1)
; CHECK:       %s3 = add i32 %s2, %phi2
; CHECK-NEXT:  -->  ({{{{}}((2 * %a) + %b),+,1}<%loop1>,+,2}<%loop2> + %v1)

entry:
  br label %loop1

loop1:
  %phi1 = phi i32 [ %a, %entry ], [ %phi1.inc, %loop1 ]
  %phi1.inc = add i32 %phi1, 1
  %cond1 = icmp ult i32 %phi1, %c
  br i1 %cond1, label %loop1, label %loop2

loop2:
  %phi2 = phi i32 [ %a, %loop1 ], [ %phi2.inc, %loop2 ]
  %phi2.inc = add i32 %phi2, 2
  %v1 = load i32, ptr %p
  %s1 = add i32 %phi1, %v1
  %s2 = add i32 %s1, %b
  %s3 = add i32 %s2, %phi2
  %cond2 = icmp ult i32 %s3, %c
  br i1 %cond2, label %loop2, label %exit

exit:

  ret void
}

; Another mix of previous use cases that demonstrates that incorrect picking of
; a loop for a recurrence may cause a crash of SCEV analysis.
define void @test_04() {

; CHECK-LABEL: Classifying expressions for: @test_04
; CHECK:       %tmp = phi i64 [ 2, %bb ], [ %tmp4, %bb3 ]
; CHECK-NEXT:  -->  {2,+,1}<nuw><nsw><%loop1>
; CHECK:       %tmp2 = trunc i64 %tmp to i32
; CHECK-NEXT:  -->  {2,+,1}<%loop1>
; CHECK:       %tmp4 = add nuw nsw i64 %tmp, 1
; CHECK-NEXT:  -->  {3,+,1}<nuw><%loop1>
; CHECK:       %tmp7 = phi i64 [ %tmp15, %loop2 ], [ 2, %loop1 ]
; CHECK-NEXT:  -->  {2,+,1}<nuw><nsw><%loop2> U: [2,9223372036854775807) S: [2,9223372036854775807)
; CHECK:       %tmp9 = sext i8 %tmp8 to i64
; CHECK-NEXT:  -->  (sext i8 %tmp8 to i64) U: [-128,128) S: [-128,128)
; CHECK:       %tmp10 = sub i64 %tmp9, %tmp7
; CHECK-NEXT:  -->  ((sext i8 %tmp8 to i64) + {-2,+,-1}<nw><%loop2>) U: [9223372036854775682,126) S: [9223372036854775682,126)
; CHECK:       %tmp11 = add i64 %tmp10, undef
; CHECK-NEXT:  -->  ((sext i8 %tmp8 to i64) + {(-2 + undef),+,-1}<nw><%loop2>)
; CHECK:       %tmp13 = trunc i64 %tmp11 to i32
; CHECK-NEXT:  -->  ((sext i8 %tmp8 to i32) + {(-2 + (trunc i64 undef to i32)),+,-1}<%loop2>)
; CHECK:       %tmp14 = sub i32 %tmp13, %tmp2
; `{{[{][{]}}` is the ugliness needed to match `{{`
; CHECK-NEXT:  -->  ((sext i8 %tmp8 to i32) + {{[{][{]}}(-4 + (trunc i64 undef to i32)),+,-1}<%loop1>,+,-1}<%loop2>)
; CHECK:       %tmp15 = add nuw nsw i64 %tmp7, 1
; CHECK-NEXT:  -->  {3,+,1}<nuw><nsw><%loop2>

bb:
  br label %loop1

loop1:
  %tmp = phi i64 [ 2, %bb ], [ %tmp4, %bb3 ]
  %tmp2 = trunc i64 %tmp to i32
  br i1 undef, label %loop2, label %bb3

bb3:
  %tmp4 = add nuw nsw i64 %tmp, 1
  br label %loop1

bb5:
  ret void

loop2:
  %tmp7 = phi i64 [ %tmp15, %loop2 ], [ 2, %loop1 ]
  %tmp8 = load i8, ptr addrspace(1) undef, align 1
  %tmp9 = sext i8 %tmp8 to i64
  %tmp10 = sub i64 %tmp9, %tmp7
  %tmp11 = add i64 %tmp10, undef
  %tmp13 = trunc i64 %tmp11 to i32
  %tmp14 = sub i32 %tmp13, %tmp2
  %tmp15 = add nuw nsw i64 %tmp7, 1
  %tmp16 = icmp slt i64 %tmp15, %tmp
  br i1 %tmp16, label %loop2, label %bb5
}

@A = weak global [1000 x i32] zeroinitializer, align 32

; Demonstrate a situation when we can add two recs with different degrees from
; the same loop.
define void @test_05(i32 %N) {

; CHECK-LABEL: Classifying expressions for: @test_05
; CHECK:       %SQ = mul i32 %i.0, %i.0
; CHECK-NEXT:  -->  {4,+,5,+,2}<%bb3>
; CHECK:       %tmp4 = mul i32 %i.0, 2
; CHECK-NEXT:  -->  {4,+,2}<nuw><nsw><%bb3>
; CHECK:       %tmp5 = sub i32 %SQ, %tmp4
; CHECK-NEXT:  -->  {0,+,3,+,2}<%bb3>

entry:
        %"alloca point" = bitcast i32 0 to i32           ; <i32> [#uses=0]
        br label %bb3

bb:             ; preds = %bb3
        %tmp = getelementptr [1000 x i32], ptr @A, i32 0, i32 %i.0          ; <ptr> [#uses=1]
        store i32 123, ptr %tmp
        %tmp2 = add i32 %i.0, 1         ; <i32> [#uses=1]
        br label %bb3

bb3:            ; preds = %bb, %entry
        %i.0 = phi i32 [ 2, %entry ], [ %tmp2, %bb ]            ; <i32> [#uses=3]
        %SQ = mul i32 %i.0, %i.0
        %tmp4 = mul i32 %i.0, 2
        %tmp5 = sub i32 %SQ, %tmp4
        %tmp3 = icmp sle i32 %tmp5, 9999          ; <i1> [#uses=1]
        br i1 %tmp3, label %bb, label %bb5

bb5:            ; preds = %bb3
        br label %return

return:         ; preds = %bb5
        ret void
}

; Check that we can add Phis from different loops with different nesting, nested
; loop comes first.
define void @test_06() {

; CHECK-LABEL: Classifying expressions for: @test_06
; CHECK:       %s1 = add i32 %phi1, %phi2
; CHECK-NEXT:  -->  {{{{}}30,+,1}<nuw><nsw><%loop1>,+,2}<nw><%loop2>
; CHECK:       %s2 = add i32 %phi2, %phi1
; CHECK-NEXT:  -->  {{{{}}30,+,1}<nuw><nsw><%loop1>,+,2}<nw><%loop2>
; CHECK:       %s3 = add i32 %phi1, %phi3
; CHECK-NEXT:  -->  {{{{}}40,+,1}<nuw><nsw><%loop1>,+,3}<nw><%loop3>
; CHECK:       %s4 = add i32 %phi3, %phi1
; CHECK-NEXT:  -->  {{{{}}40,+,1}<nuw><nsw><%loop1>,+,3}<nw><%loop3>
; CHECK:       %s5 = add i32 %phi2, %phi3
; CHECK-NEXT:  -->  {{{{}}50,+,2}<nuw><nsw><%loop2>,+,3}<nw><%loop3>
; CHECK:       %s6 = add i32 %phi3, %phi2
; CHECK-NEXT:  -->  {{{{}}50,+,2}<nuw><nsw><%loop2>,+,3}<nw><%loop3>

entry:
  br label %loop1

loop1:
  %phi1 = phi i32 [ 10, %entry ], [ %phi1.inc, %loop1.exit ]
  br label %loop2

loop2:
  %phi2 = phi i32 [ 20, %loop1 ], [ %phi2.inc, %loop2 ]
  %phi2.inc = add i32 %phi2, 2
  %cond2 = icmp ult i32 %phi2.inc, 1000
  br i1 %cond2, label %loop2, label %loop1.exit

loop1.exit:
  %phi1.inc = add i32 %phi1, 1
  %cond1 = icmp ult i32 %phi1.inc, 1000
  br i1 %cond1, label %loop1, label %loop3

loop3:
  %phi3 = phi i32 [ 30, %loop1.exit ], [ %phi3.inc, %loop3 ]
  %phi3.inc = add i32 %phi3, 3
  %cond3 = icmp ult i32 %phi3.inc, 1000
  br i1 %cond3, label %loop3, label %exit

exit:
  %s1 = add i32 %phi1, %phi2
  %s2 = add i32 %phi2, %phi1
  %s3 = add i32 %phi1, %phi3
  %s4 = add i32 %phi3, %phi1
  %s5 = add i32 %phi2, %phi3
  %s6 = add i32 %phi3, %phi2
  ret void
}

; Check that we can add Phis from different loops with different nesting, nested
; loop comes second.
define void @test_07() {

; CHECK-LABEL: Classifying expressions for: @test_07
; CHECK:       %s1 = add i32 %phi1, %phi2
; CHECK-NEXT:  -->  {{{{}}30,+,1}<nuw><nsw><%loop1>,+,2}<nw><%loop2>
; CHECK:       %s2 = add i32 %phi2, %phi1
; CHECK-NEXT:  -->  {{{{}}30,+,1}<nuw><nsw><%loop1>,+,2}<nw><%loop2>
; CHECK:       %s3 = add i32 %phi1, %phi3
; CHECK-NEXT:  -->  {{{{}}40,+,3}<nuw><nsw><%loop3>,+,1}<nw><%loop1>
; CHECK:       %s4 = add i32 %phi3, %phi1
; CHECK-NEXT:  -->  {{{{}}40,+,3}<nuw><nsw><%loop3>,+,1}<nw><%loop1>
; CHECK:       %s5 = add i32 %phi2, %phi3
; CHECK-NEXT:  -->  {{{{}}50,+,3}<nuw><nsw><%loop3>,+,2}<nw><%loop2>
; CHECK:       %s6 = add i32 %phi3, %phi2
; CHECK-NEXT:  -->  {{{{}}50,+,3}<nuw><nsw><%loop3>,+,2}<nw><%loop2>

entry:
  br label %loop3

loop3:
  %phi3 = phi i32 [ 30, %entry ], [ %phi3.inc, %loop3 ]
  %phi3.inc = add i32 %phi3, 3
  %cond3 = icmp ult i32 %phi3.inc, 1000
  br i1 %cond3, label %loop3, label %loop1

loop1:
  %phi1 = phi i32 [ 10, %loop3 ], [ %phi1.inc, %loop1.exit ]
  br label %loop2

loop2:
  %phi2 = phi i32 [ 20, %loop1 ], [ %phi2.inc, %loop2 ]
  %phi2.inc = add i32 %phi2, 2
  %cond2 = icmp ult i32 %phi2.inc, 1000
  br i1 %cond2, label %loop2, label %loop1.exit

loop1.exit:
  %phi1.inc = add i32 %phi1, 1
  %cond1 = icmp ult i32 %phi1.inc, 1000
  br i1 %cond1, label %exit, label %loop1

exit:
  %s1 = add i32 %phi1, %phi2
  %s2 = add i32 %phi2, %phi1
  %s3 = add i32 %phi1, %phi3
  %s4 = add i32 %phi3, %phi1
  %s5 = add i32 %phi2, %phi3
  %s6 = add i32 %phi3, %phi2
  ret void
}

; Make sure that a complicated Phi does not get folded with rec's start value
; of a loop which is above.
define void @test_08() {

; CHECK-LABEL: Classifying expressions for: @test_08
; CHECK:       %tmp11 = add i64 %iv.2.2, %iv.2.1
; CHECK-NEXT:  -->  ({0,+,-1}<nuw><nsw><%loop_2> + %iv.2.1)
; CHECK:       %tmp12 = trunc i64 %tmp11 to i32
; CHECK-NEXT:  -->  ((trunc i64 %iv.2.1 to i32) + {0,+,-1}<%loop_2>)
; CHECK:       %tmp14 = mul i32 %tmp12, %tmp7
; CHECK-NEXT:  -->  (((trunc i64 %iv.2.1 to i32) + {0,+,-1}<%loop_2>) * {-1,+,-1}<%loop_1>)
; CHECK:       %tmp16 = mul i64 %iv.2.1, %iv.1.1
; CHECK-NEXT:  -->  ({2,+,1}<nuw><nsw><%loop_1> * %iv.2.1)

entry:
  br label %loop_1

loop_1:
  %iv.1.1 = phi i64 [ 2, %entry ], [ %iv.1.1.next, %loop_1_back_branch ]
  %iv.1.2 = phi i32 [ -1, %entry ], [ %iv.1.2.next, %loop_1_back_branch ]
  br label %loop_1_exit

dead:
  br label %loop_1_exit

loop_1_exit:
  %tmp5 = icmp sgt i64 %iv.1.1, 2
  br i1 %tmp5, label %loop_2_preheader, label %loop_1_back_branch

loop_1_back_branch:
  %iv.1.1.next = add nuw nsw i64 %iv.1.1, 1
  %iv.1.2.next = add nsw i32 %iv.1.2, 1
  br label %loop_1

loop_2_preheader:
  %tmp6 = sub i64 1, %iv.1.1
  %tmp7 = trunc i64 %tmp6 to i32
  br label %loop_2

loop_2:
  %iv.2.1 = phi i64 [ 0, %loop_2_preheader ], [ %tmp16, %loop_2 ]
  %iv.2.2 = phi i64 [ 0, %loop_2_preheader ], [ %iv.2.2.next, %loop_2 ]
  %iv.2.3 = phi i64 [ 2, %loop_2_preheader ], [ %iv.2.3.next, %loop_2 ]
  %tmp11 = add i64 %iv.2.2, %iv.2.1
  %tmp12 = trunc i64 %tmp11 to i32
  %tmp14 = mul i32 %tmp12, %tmp7
  %tmp16 = mul i64 %iv.2.1, %iv.1.1
  %iv.2.3.next = add nuw nsw i64 %iv.2.3, 1
  %iv.2.2.next = add nsw i64 %iv.2.2, -1
  %tmp17 = icmp slt i64 %iv.2.3.next, %iv.1.1
  br i1 %tmp17, label %loop_2, label %exit

exit:
  %tmp10 = add i32 %iv.1.2, 3
  ret void
}

define i64 @test_09(i32 %param) {

; CHECK-LABEL: Classifying expressions for: @test_09
; CHECK:       %iv1 = phi i64 [ %iv1.next, %guarded ], [ 0, %outer.loop ]
; CHECK-NEXT:    -->  {0,+,1}<nuw><nsw><%loop1>
; CHECK:       %iv1.trunc = trunc i64 %iv1 to i32
; CHECK-NEXT:    -->  {0,+,1}<%loop1>
; CHECK:       %iv1.next = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:    -->  {1,+,1}<nuw><nsw><%loop1>
; CHECK:       %iv2 = phi i32 [ %iv2.next, %loop2 ], [ %param, %loop2.preheader ]
; CHECK-NEXT:    -->  {%param,+,1}<%loop2>
; CHECK:       %iv2.next = add i32 %iv2, 1
; CHECK-NEXT:    -->  {(1 + %param),+,1}<%loop2>
; CHECK:       %iv2.ext = sext i32 %iv2.next to i64
; CHECK-NEXT:    -->  (sext i32 {(1 + %param),+,1}<%loop2> to i64)
; CHECK:       %ret = mul i64 %iv1, %iv2.ext
; CHECK-NEXT:    -->  ((sext i32 {(1 + %param),+,1}<%loop2> to i64) * {0,+,1}<nuw><nsw><%loop1>)

entry:
  br label %outer.loop

outer.loop:                                 ; preds = %loop2.exit, %entry
  br label %loop1

loop1:                                           ; preds = %guarded, %outer.loop
  %iv1 = phi i64 [ %iv1.next, %guarded ], [ 0, %outer.loop ]
  %iv1.trunc = trunc i64 %iv1 to i32
  %cond1 = icmp ult i64 %iv1, 100
  br i1 %cond1, label %guarded, label %deopt

guarded:                                          ; preds = %loop1
  %iv1.next = add nuw nsw i64 %iv1, 1
  %tmp16 = icmp slt i32 %iv1.trunc, 2
  br i1 %tmp16, label %loop1, label %loop2.preheader

deopt:                                            ; preds = %loop1
  unreachable

loop2.preheader:                                 ; preds = %guarded
  br label %loop2

loop2:                                           ; preds = %loop2, %loop2.preheader
  %iv2 = phi i32 [ %iv2.next, %loop2 ], [ %param, %loop2.preheader ]
  %iv2.next = add i32 %iv2, 1
  %cond2 = icmp slt i32 %iv2, %iv1.trunc
  br i1 %cond2, label %loop2, label %exit

exit:                                          ; preds = %loop2.exit
  %iv2.ext = sext i32 %iv2.next to i64
  %ret = mul i64 %iv1, %iv2.ext
  ret i64 %ret
}

define i64 @test_10(i32 %param) {

; CHECK-LABEL: Classifying expressions for: @test_10
; CHECK:       %uncle = phi i64 [ %uncle.outer.next, %uncle.loop.backedge ], [ 0, %outer.loop ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%uncle.loop>
; CHECK:       %iv1 = phi i64 [ %iv1.next, %guarded ], [ 0, %uncle.loop ]
; CHECK-NEXT:  -->  {0,+,1}<nuw><nsw><%loop1>
; CHECK:       %iv1.trunc = trunc i64 %iv1 to i32
; CHECK-NEXT:  -->  {0,+,1}<%loop1>
; CHECK:       %iv1.next = add nuw nsw i64 %iv1, 1
; CHECK-NEXT:  -->  {1,+,1}<nuw><nsw><%loop1>
; CHECK:       %uncle.outer.next = add i64 %uncle, 1
; CHECK-NEXT:  -->  {1,+,1}<nuw><nsw><%uncle.loop>
; CHECK:       %iv2 = phi i32 [ %iv2.next, %loop2 ], [ %param, %loop2.preheader ]
; CHECK-NEXT:  -->  {%param,+,1}<%loop2>
; CHECK:       %iv2.next = add i32 %iv2, 1
; CHECK-NEXT:  -->  {(1 + %param),+,1}<%loop2>
; CHECK:       %iv2.ext = sext i32 %iv2.next to i64
; CHECK-NEXT:  -->  (sext i32 {(1 + %param),+,1}<%loop2> to i64)
; CHECK:       %ret = mul i64 %iv1, %iv2.ext
; CHECK-NEXT:  -->  ((sext i32 {(1 + %param),+,1}<%loop2> to i64) * {0,+,1}<nuw><nsw><%loop1>)

entry:
  br label %outer.loop

outer.loop:                                       ; preds = %entry
  br label %uncle.loop

uncle.loop:                                       ; preds = %uncle.loop.backedge, %outer.loop
  %uncle = phi i64 [ %uncle.outer.next, %uncle.loop.backedge ], [ 0, %outer.loop ]
  br label %loop1

loop1:                                            ; preds = %guarded, %uncle.loop
  %iv1 = phi i64 [ %iv1.next, %guarded ], [ 0, %uncle.loop ]
  %iv1.trunc = trunc i64 %iv1 to i32
  %cond1 = icmp ult i64 %iv1, 100
  br i1 %cond1, label %guarded, label %deopt

guarded:                                          ; preds = %loop1
  %iv1.next = add nuw nsw i64 %iv1, 1
  %tmp16 = icmp slt i32 %iv1.trunc, 2
  br i1 %tmp16, label %loop1, label %uncle.loop.backedge

uncle.loop.backedge:                              ; preds = %guarded
  %uncle.outer.next = add i64 %uncle, 1
  %cond.uncle = icmp ult i64 %uncle, 120
  br i1 %cond.uncle, label %loop2.preheader, label %uncle.loop

deopt:                                            ; preds = %loop1
  unreachable

loop2.preheader:                                  ; preds = %uncle.loop.backedge
  br label %loop2

loop2:                                            ; preds = %loop2, %loop2.preheader
  %iv2 = phi i32 [ %iv2.next, %loop2 ], [ %param, %loop2.preheader ]
  %iv2.next = add i32 %iv2, 1
  %cond2 = icmp slt i32 %iv2, %iv1.trunc
  br i1 %cond2, label %loop2, label %exit

exit:                                             ; preds = %loop2
  %iv2.ext = sext i32 %iv2.next to i64
  %ret = mul i64 %iv1, %iv2.ext
  ret i64 %ret
}
