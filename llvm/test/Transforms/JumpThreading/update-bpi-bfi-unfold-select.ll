; RUN: opt -passes="require<branch-prob>,require<block-freq>,jump-threading" -S < %s | FileCheck %s

@d = global i8 0, align 1
@c = global i32 0, align 4
@b = global i1 false, align 2
@f = global i32 0, align 4
@e = global i32 0, align 4
@a = global i32 0, align 4

; CHECK-LABEL: @test
define i32 @test() !prof !0 {
bb:
  %i = load i8, ptr @d, align 1
  %i1 = add i8 %i, 1
  store i8 %i1, ptr @d, align 1
  %i2 = icmp eq i8 %i1, 0
  br i1 %i2, label %bb4, label %bb15, !prof !1

bb4:
  %i5 = load i1, ptr @b, align 2
  %i6 = select i1 %i5, i32 0, i32 2
  %i7 = load i32, ptr @c, align 4
  %i8 = xor i32 %i7, %i6
  store i32 %i8, ptr @c, align 4
  %i9 = icmp eq i32 %i7, %i6
  br i1 %i9, label %bb13, label %bb10, !prof !1

bb10:
  %i11 = load i32, ptr @f, align 4
  %i12 = or i32 %i11, 9
  store i32 %i12, ptr @f, align 4
  br label %bb18

bb13:
  store i1 true, ptr @b, align 2
  br label %bb18

bb15:
  store i32 0, ptr @c, align 4
  %i16 = load i32, ptr @e, align 4
  %i17 = or i32 %i16, 9
  store i32 %i17, ptr @e, align 4
  %i19.pre = load i1, ptr @b, align 2
  br label %bb18

bb18:
  %i19 = phi i1 [ %i5, %bb10 ], [ true, %bb13 ], [ %i19.pre, %bb15 ]
  %i20 = select i1 %i19, i32 0, i32 2, !prof !2
  store i32 %i20, ptr @a, align 4
  ret i32 0
}

!0 = !{!"function_entry_count", i64 10}
!1 = !{!"branch_weights", i32 5, i32 5}
!2 = !{!"branch_weights", i32 0, i32 10}

; CHECK: br i1 %cond.fr{{.*}}, label %{{.*}}, label %{{.*}}, !prof ![[MD:[0-9]+]]
; CHECK: ![[MD]] = !{!"branch_weights", i32 0, i32 -2147483648}
