; RUN: opt -S -passes='function(lower-expect),strip-dead-prototypes' -likely-branch-weight=2147483647 -unlikely-branch-weight=1 < %s | FileCheck %s

; The C case
; if (__builtin_expect_with_probability(((a0 == 1) || (a1 == 1) || (a2 == 1)), 1, 0))
; For the above case, all 3 branches should be annotated
; which should be equivalent to if (__builtin_expect(((a0 == 1) || (a1 == 1) || (a2 == 1)), 0))

; The C case
; if (__builtin_expect_with_probability(((a0 == 1) || (a1 == 1) || (a2 == 1)), 1, 1))
; For the above case, we do not have enough information, so only the last branch could be annotated
; which should be equivalent to if (__builtin_expect(((a0 == 1) || (a1 == 1) || (a2 == 1)), 1))

declare void @foo()

declare i64 @llvm.expect.i64(i64, i64) nounwind readnone
declare i64 @llvm.expect.with.probability.i64(i64, i64, double) nounwind readnone

; CHECK-LABEL: @test1_expect_1(
; CHECK: block0:
; CHECK-NOT: prof
; CHECK: block1:
; CHECK-NOT: prof
; CHECK: block3:
; CHECK: br i1 %tobool, label %block4, label %block5, !prof !0
define void @test1_expect_1(i8 %a0, i8 %a1, i8 %a2) {
block0:
  %c0 = icmp eq i8 %a0, 1
  br i1 %c0, label %block3, label %block1

block1:
  %c1 = icmp eq i8 %a1, 1
  br i1 %c1, label %block3, label %block2

block2:
  %c2 = icmp eq i8 %a2, 1
  br label %block3

block3:
  %cond0 = phi i1 [ true, %block0 ], [ true, %block1 ], [ %c2, %block2 ]
  %cond1 = zext i1 %cond0 to i32
  %cond2 = sext i32 %cond1 to i64
  %expval = call i64 @llvm.expect.i64(i64 %cond2, i64 1)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %block4, label %block5

block4:
  call void @foo()
  br label %block5

block5:
  ret void
}

; should have exactly the same behavior as test1
; CHECK-LABEL: @test2_expect_with_prob_1_1(
; CHECK: block0:
; CHECK-NOT: prof
; CHECK: block1:
; CHECK-NOT: prof
; CHECK: block3:
; CHECK: br i1 %tobool, label %block4, label %block5, !prof !0
define void @test2_expect_with_prob_1_1(i8 %a0, i8 %a1, i8 %a2) {
block0:
  %c0 = icmp eq i8 %a0, 1
  br i1 %c0, label %block3, label %block1

block1:
  %c1 = icmp eq i8 %a1, 1
  br i1 %c1, label %block3, label %block2

block2:
  %c2 = icmp eq i8 %a2, 1
  br label %block3

block3:
  %cond0 = phi i1 [ true, %block0 ], [ true, %block1 ], [ %c2, %block2 ]
  %cond1 = zext i1 %cond0 to i32
  %cond2 = sext i32 %cond1 to i64
  %expval = call i64 @llvm.expect.with.probability.i64(i64 %cond2, i64 1, double 1.0)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %block4, label %block5

block4:
  call void @foo()
  br label %block5

block5:
  ret void
}

; should have exactly the same behavior as test1
; CHECK-LABEL: @test3_expect_with_prob_0_0(
; CHECK: block0:
; CHECK-NOT: prof
; CHECK: block1:
; CHECK-NOT: prof
; CHECK: block3:
; CHECK: br i1 %tobool, label %block4, label %block5, !prof !0
define void @test3_expect_with_prob_0_0(i8 %a0, i8 %a1, i8 %a2) {
block0:
  %c0 = icmp eq i8 %a0, 1
  br i1 %c0, label %block3, label %block1

block1:
  %c1 = icmp eq i8 %a1, 1
  br i1 %c1, label %block3, label %block2

block2:
  %c2 = icmp eq i8 %a2, 1
  br label %block3

block3:
  %cond0 = phi i1 [ true, %block0 ], [ true, %block1 ], [ %c2, %block2 ]
  %cond1 = zext i1 %cond0 to i32
  %cond2 = sext i32 %cond1 to i64
  %expval = call i64 @llvm.expect.with.probability.i64(i64 %cond2, i64 0, double 0.0)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %block4, label %block5

block4:
  call void @foo()
  br label %block5

block5:
  ret void
}

; CHECK-LABEL: @test4_expect_0(
; CHECK: block0:
; CHECK: br i1 %c0, label %block3, label %block1, !prof !1
; CHECK: block1:
; CHECK: br i1 %c1, label %block3, label %block2, !prof !1
; CHECK: block3:
; CHECK: br i1 %tobool, label %block4, label %block5, !prof !1
define void @test4_expect_0(i8 %a0, i8 %a1, i8 %a2) {
block0:
  %c0 = icmp eq i8 %a0, 1
  br i1 %c0, label %block3, label %block1

block1:
  %c1 = icmp eq i8 %a1, 1
  br i1 %c1, label %block3, label %block2

block2:
  %c2 = icmp eq i8 %a2, 1
  br label %block3

block3:
  %cond0 = phi i1 [ true, %block0 ], [ true, %block1 ], [ %c2, %block2 ]
  %cond1 = zext i1 %cond0 to i32
  %cond2 = sext i32 %cond1 to i64
  %expval = call i64 @llvm.expect.i64(i64 %cond2, i64 0)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %block4, label %block5

block4:
  call void @foo()
  br label %block5

block5:
  ret void
}

; should have exactly the same behavior as test4
; CHECK-LABEL: @test5_expect_with_prob_1_0(
; CHECK: block0:
; CHECK: br i1 %c0, label %block3, label %block1, !prof !1
; CHECK: block1:
; CHECK: br i1 %c1, label %block3, label %block2, !prof !1
; CHECK: block3:
; CHECK: br i1 %tobool, label %block4, label %block5, !prof !1
define void @test5_expect_with_prob_1_0(i8 %a0, i8 %a1, i8 %a2) {
block0:
  %c0 = icmp eq i8 %a0, 1
  br i1 %c0, label %block3, label %block1

block1:
  %c1 = icmp eq i8 %a1, 1
  br i1 %c1, label %block3, label %block2

block2:
  %c2 = icmp eq i8 %a2, 1
  br label %block3

block3:
  %cond0 = phi i1 [ true, %block0 ], [ true, %block1 ], [ %c2, %block2 ]
  %cond1 = zext i1 %cond0 to i32
  %cond2 = sext i32 %cond1 to i64
  %expval = call i64 @llvm.expect.with.probability.i64(i64 %cond2, i64 1, double 0.0)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %block4, label %block5

block4:
  call void @foo()
  br label %block5

block5:
  ret void
}

; should have exactly the same behavior as test4
; CHECK-LABEL: @test6_expect_with_prob_0_1(
; CHECK: block0:
; CHECK: br i1 %c0, label %block3, label %block1, !prof !1
; CHECK: block1:
; CHECK: br i1 %c1, label %block3, label %block2, !prof !1
; CHECK: block3:
; CHECK: br i1 %tobool, label %block4, label %block5, !prof !1
define void @test6_expect_with_prob_0_1(i8 %a0, i8 %a1, i8 %a2) {
block0:
  %c0 = icmp eq i8 %a0, 1
  br i1 %c0, label %block3, label %block1

block1:
  %c1 = icmp eq i8 %a1, 1
  br i1 %c1, label %block3, label %block2

block2:
  %c2 = icmp eq i8 %a2, 1
  br label %block3

block3:
  %cond0 = phi i1 [ true, %block0 ], [ true, %block1 ], [ %c2, %block2 ]
  %cond1 = zext i1 %cond0 to i32
  %cond2 = sext i32 %cond1 to i64
  %expval = call i64 @llvm.expect.with.probability.i64(i64 %cond2, i64 0, double 1.0)
  %tobool = icmp ne i64 %expval, 0
  br i1 %tobool, label %block4, label %block5

block4:
  call void @foo()
  br label %block5

block5:
  ret void
}

; CHECK: !0 = !{!"branch_weights", i32 2147483647, i32 1}
; CHECK: !1 = !{!"branch_weights", i32 1, i32 2147483647}
