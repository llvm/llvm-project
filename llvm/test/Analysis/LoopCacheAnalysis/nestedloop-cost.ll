; RUN: opt < %s -passes='print<loop-cache-cost>' -disable-output 2>&1 | FileCheck %s
;
; In a deeply nested loop (such as the one in this test), it is possible for
; LoopCacheAnalysis to crash when computing its cost. In the non-consecutive case,
; computeRefCost() calls getExtendedType(), which doubles the SCEV width at each
; step. With many dimensions (20 in this test), it is possible for the width to
; exceed MAX_INT_BITS and asserting in IntegerType::get().
; Instead of asserting, we return CacheCostTy::getInvalid() in the case where we
; exceed MAX_INT_BITS.
;
; CHECK-LABEL: 'test'
; CHECK-NEXT: Loop 'l1' has cost = Invalid
; CHECK-NEXT: Loop 'l2' has cost = 137438953472
; CHECK-NEXT: Loop 'l3' has cost = 68719476736
; CHECK-NEXT: Loop 'l4' has cost = 34359738368
; CHECK-NEXT: Loop 'l5' has cost = 17179869184
; CHECK-NEXT: Loop 'l6' has cost = 8589934592
; CHECK-NEXT: Loop 'l7' has cost = 4294967296
; CHECK-NEXT: Loop 'l8' has cost = 2147483648
; CHECK-NEXT: Loop 'l9' has cost = 1073741824
; CHECK-NEXT: Loop 'l10' has cost = 536870912
; CHECK-NEXT: Loop 'l11' has cost = 268435456
; CHECK-NEXT: Loop 'l12' has cost = 134217728
; CHECK-NEXT: Loop 'l13' has cost = 67108864
; CHECK-NEXT: Loop 'l14' has cost = 33554432
; CHECK-NEXT: Loop 'l15' has cost = 16777216
; CHECK-NEXT: Loop 'l16' has cost = 8388608
; CHECK-NEXT: Loop 'l17' has cost = 4194304
; CHECK-NEXT: Loop 'l18' has cost = 2097152
; CHECK-NEXT: Loop 'l19' has cost = 1048576
; CHECK-NEXT: Loop 'l20' has cost = 1048576

@A = global [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x i32]]]]]]]]]]]]]]]]]]]] zeroinitializer

define void @test() {
entry:
  br label %l1

l1:
  %i1 = phi i32 [ 0, %entry ], [ %i1n, %l1e ]
  br label %l2
l2:
  %i2 = phi i32 [ 0, %l1 ], [ %i2n, %l2e ]
  br label %l3
l3:
  %i3 = phi i32 [ 0, %l2 ], [ %i3n, %l3e ]
  br label %l4
l4:
  %i4 = phi i32 [ 0, %l3 ], [ %i4n, %l4e ]
  br label %l5
l5:
  %i5 = phi i32 [ 0, %l4 ], [ %i5n, %l5e ]
  br label %l6
l6:
  %i6 = phi i32 [ 0, %l5 ], [ %i6n, %l6e ]
  br label %l7
l7:
  %i7 = phi i32 [ 0, %l6 ], [ %i7n, %l7e ]
  br label %l8
l8:
  %i8 = phi i32 [ 0, %l7 ], [ %i8n, %l8e ]
  br label %l9
l9:
  %i9 = phi i32 [ 0, %l8 ], [ %i9n, %l9e ]
  br label %l10
l10:
  %i10 = phi i32 [ 0, %l9 ], [ %i10n, %l10e ]
  br label %l11
l11:
  %i11 = phi i32 [ 0, %l10 ], [ %i11n, %l11e ]
  br label %l12
l12:
  %i12 = phi i32 [ 0, %l11 ], [ %i12n, %l12e ]
  br label %l13
l13:
  %i13 = phi i32 [ 0, %l12 ], [ %i13n, %l13e ]
  br label %l14
l14:
  %i14 = phi i32 [ 0, %l13 ], [ %i14n, %l14e ]
  br label %l15
l15:
  %i15 = phi i32 [ 0, %l14 ], [ %i15n, %l15e ]
  br label %l16
l16:
  %i16 = phi i32 [ 0, %l15 ], [ %i16n, %l16e ]
  br label %l17
l17:
  %i17 = phi i32 [ 0, %l16 ], [ %i17n, %l17e ]
  br label %l18
l18:
  %i18 = phi i32 [ 0, %l17 ], [ %i18n, %l18e ]
  br label %l19
l19:
  %i19 = phi i32 [ 0, %l18 ], [ %i19n, %l19e ]
  br label %l20

l20:
  %i20 = phi i32 [ 0, %l19 ], [ %i20n, %l20 ]
  %gep = getelementptr inbounds [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x [2 x i32]]]]]]]]]]]]]]]]]]],
           ptr @A,
           i32 %i1,  i32 %i2,  i32 %i3,  i32 %i4,  i32 %i5,
           i32 %i6,  i32 %i7,  i32 %i8,  i32 %i9,  i32 %i10,
           i32 %i11, i32 %i12, i32 %i13, i32 %i14, i32 %i15,
           i32 %i16, i32 %i17, i32 %i18, i32 %i19, i32 %i20
  store i32 1, ptr %gep, align 4
  %i20n = add i32 %i20, 1
  %c20 = icmp slt i32 %i20n, 2
  br i1 %c20, label %l20, label %l19e

l19e:
  %i19n = add i32 %i19, 1
  %c19 = icmp slt i32 %i19n, 2
  br i1 %c19, label %l19, label %l18e
l18e:
  %i18n = add i32 %i18, 1
  %c18 = icmp slt i32 %i18n, 2
  br i1 %c18, label %l18, label %l17e
l17e:
  %i17n = add i32 %i17, 1
  %c17 = icmp slt i32 %i17n, 2
  br i1 %c17, label %l17, label %l16e
l16e:
  %i16n = add i32 %i16, 1
  %c16 = icmp slt i32 %i16n, 2
  br i1 %c16, label %l16, label %l15e
l15e:
  %i15n = add i32 %i15, 1
  %c15 = icmp slt i32 %i15n, 2
  br i1 %c15, label %l15, label %l14e
l14e:
  %i14n = add i32 %i14, 1
  %c14 = icmp slt i32 %i14n, 2
  br i1 %c14, label %l14, label %l13e
l13e:
  %i13n = add i32 %i13, 1
  %c13 = icmp slt i32 %i13n, 2
  br i1 %c13, label %l13, label %l12e
l12e:
  %i12n = add i32 %i12, 1
  %c12 = icmp slt i32 %i12n, 2
  br i1 %c12, label %l12, label %l11e
l11e:
  %i11n = add i32 %i11, 1
  %c11 = icmp slt i32 %i11n, 2
  br i1 %c11, label %l11, label %l10e
l10e:
  %i10n = add i32 %i10, 1
  %c10 = icmp slt i32 %i10n, 2
  br i1 %c10, label %l10, label %l9e
l9e:
  %i9n = add i32 %i9, 1
  %c9 = icmp slt i32 %i9n, 2
  br i1 %c9, label %l9, label %l8e
l8e:
  %i8n = add i32 %i8, 1
  %c8 = icmp slt i32 %i8n, 2
  br i1 %c8, label %l8, label %l7e
l7e:
  %i7n = add i32 %i7, 1
  %c7 = icmp slt i32 %i7n, 2
  br i1 %c7, label %l7, label %l6e
l6e:
  %i6n = add i32 %i6, 1
  %c6 = icmp slt i32 %i6n, 2
  br i1 %c6, label %l6, label %l5e
l5e:
  %i5n = add i32 %i5, 1
  %c5 = icmp slt i32 %i5n, 2
  br i1 %c5, label %l5, label %l4e
l4e:
  %i4n = add i32 %i4, 1
  %c4 = icmp slt i32 %i4n, 2
  br i1 %c4, label %l4, label %l3e
l3e:
  %i3n = add i32 %i3, 1
  %c3 = icmp slt i32 %i3n, 2
  br i1 %c3, label %l3, label %l2e
l2e:
  %i2n = add i32 %i2, 1
  %c2 = icmp slt i32 %i2n, 2
  br i1 %c2, label %l2, label %l1e
l1e:
  %i1n = add i32 %i1, 1
  %c1 = icmp slt i32 %i1n, 2
  br i1 %c1, label %l1, label %exit

exit:
  ret void
}
