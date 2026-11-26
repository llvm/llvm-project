; RUN: opt -passes=indvars -S %s | FileCheck %s
;
; A chain of loops where each loop starts from the exit value of the previous
; one. Proving facts about the start values used to recurse through the whole
; chain from isKnownViaNonRecursiveReasoning, which took exponential time in
; the number of loops. Check that this finishes quickly.

define i64 @f(i1 %b) {
; CHECK-LABEL: define i64 @f(
; CHECK:         ret i64
entry:
  %k = zext i1 %b to i64
  br label %pre0
pre0:
  %r0 = urem i64 %k, 6
  %n0 = add nuw nsw i64 %r0, 1
  br label %loop0
loop0:
  %x0 = phi i64 [ %k, %pre0 ], [ %x0.next, %latch0 ]
  %i0 = phi i64 [ 0, %pre0 ], [ %i0.next, %latch0 ]
  %x0.next = add i64 %x0, %k
  %ov0 = icmp ult i64 %x0.next, %x0
  br i1 %ov0, label %trap, label %latch0
latch0:
  %i0.next = add nuw nsw i64 %i0, 1
  %c0 = icmp ne i64 %i0.next, %n0
  br i1 %c0, label %loop0, label %pre1
pre1:
  %r1 = urem i64 %x0.next, 6
  %n1 = add nuw nsw i64 %r1, 1
  br label %loop1
loop1:
  %x1 = phi i64 [ %x0.next, %pre1 ], [ %x1.next, %latch1 ]
  %i1 = phi i64 [ 0, %pre1 ], [ %i1.next, %latch1 ]
  %x1.next = add i64 %x1, %k
  %ov1 = icmp ult i64 %x1.next, %x1
  br i1 %ov1, label %trap, label %latch1
latch1:
  %i1.next = add nuw nsw i64 %i1, 1
  %c1 = icmp ne i64 %i1.next, %n1
  br i1 %c1, label %loop1, label %pre2
pre2:
  %r2 = urem i64 %x1.next, 6
  %n2 = add nuw nsw i64 %r2, 1
  br label %loop2
loop2:
  %x2 = phi i64 [ %x1.next, %pre2 ], [ %x2.next, %latch2 ]
  %i2 = phi i64 [ 0, %pre2 ], [ %i2.next, %latch2 ]
  %x2.next = add i64 %x2, %k
  %ov2 = icmp ult i64 %x2.next, %x2
  br i1 %ov2, label %trap, label %latch2
latch2:
  %i2.next = add nuw nsw i64 %i2, 1
  %c2 = icmp ne i64 %i2.next, %n2
  br i1 %c2, label %loop2, label %pre3
pre3:
  %r3 = urem i64 %x2.next, 6
  %n3 = add nuw nsw i64 %r3, 1
  br label %loop3
loop3:
  %x3 = phi i64 [ %x2.next, %pre3 ], [ %x3.next, %latch3 ]
  %i3 = phi i64 [ 0, %pre3 ], [ %i3.next, %latch3 ]
  %x3.next = add i64 %x3, %k
  %ov3 = icmp ult i64 %x3.next, %x3
  br i1 %ov3, label %trap, label %latch3
latch3:
  %i3.next = add nuw nsw i64 %i3, 1
  %c3 = icmp ne i64 %i3.next, %n3
  br i1 %c3, label %loop3, label %pre4
pre4:
  %r4 = urem i64 %x3.next, 6
  %n4 = add nuw nsw i64 %r4, 1
  br label %loop4
loop4:
  %x4 = phi i64 [ %x3.next, %pre4 ], [ %x4.next, %latch4 ]
  %i4 = phi i64 [ 0, %pre4 ], [ %i4.next, %latch4 ]
  %x4.next = add i64 %x4, %k
  %ov4 = icmp ult i64 %x4.next, %x4
  br i1 %ov4, label %trap, label %latch4
latch4:
  %i4.next = add nuw nsw i64 %i4, 1
  %c4 = icmp ne i64 %i4.next, %n4
  br i1 %c4, label %loop4, label %pre5
pre5:
  %r5 = urem i64 %x4.next, 6
  %n5 = add nuw nsw i64 %r5, 1
  br label %loop5
loop5:
  %x5 = phi i64 [ %x4.next, %pre5 ], [ %x5.next, %latch5 ]
  %i5 = phi i64 [ 0, %pre5 ], [ %i5.next, %latch5 ]
  %x5.next = add i64 %x5, %k
  %ov5 = icmp ult i64 %x5.next, %x5
  br i1 %ov5, label %trap, label %latch5
latch5:
  %i5.next = add nuw nsw i64 %i5, 1
  %c5 = icmp ne i64 %i5.next, %n5
  br i1 %c5, label %loop5, label %pre6
pre6:
  %r6 = urem i64 %x5.next, 6
  %n6 = add nuw nsw i64 %r6, 1
  br label %loop6
loop6:
  %x6 = phi i64 [ %x5.next, %pre6 ], [ %x6.next, %latch6 ]
  %i6 = phi i64 [ 0, %pre6 ], [ %i6.next, %latch6 ]
  %x6.next = add i64 %x6, %k
  %ov6 = icmp ult i64 %x6.next, %x6
  br i1 %ov6, label %trap, label %latch6
latch6:
  %i6.next = add nuw nsw i64 %i6, 1
  %c6 = icmp ne i64 %i6.next, %n6
  br i1 %c6, label %loop6, label %pre7
pre7:
  %r7 = urem i64 %x6.next, 6
  %n7 = add nuw nsw i64 %r7, 1
  br label %loop7
loop7:
  %x7 = phi i64 [ %x6.next, %pre7 ], [ %x7.next, %latch7 ]
  %i7 = phi i64 [ 0, %pre7 ], [ %i7.next, %latch7 ]
  %x7.next = add i64 %x7, %k
  %ov7 = icmp ult i64 %x7.next, %x7
  br i1 %ov7, label %trap, label %latch7
latch7:
  %i7.next = add nuw nsw i64 %i7, 1
  %c7 = icmp ne i64 %i7.next, %n7
  br i1 %c7, label %loop7, label %pre8
pre8:
  %r8 = urem i64 %x7.next, 6
  %n8 = add nuw nsw i64 %r8, 1
  br label %loop8
loop8:
  %x8 = phi i64 [ %x7.next, %pre8 ], [ %x8.next, %latch8 ]
  %i8 = phi i64 [ 0, %pre8 ], [ %i8.next, %latch8 ]
  %x8.next = add i64 %x8, %k
  %ov8 = icmp ult i64 %x8.next, %x8
  br i1 %ov8, label %trap, label %latch8
latch8:
  %i8.next = add nuw nsw i64 %i8, 1
  %c8 = icmp ne i64 %i8.next, %n8
  br i1 %c8, label %loop8, label %pre9
pre9:
  %r9 = urem i64 %x8.next, 6
  %n9 = add nuw nsw i64 %r9, 1
  br label %loop9
loop9:
  %x9 = phi i64 [ %x8.next, %pre9 ], [ %x9.next, %latch9 ]
  %i9 = phi i64 [ 0, %pre9 ], [ %i9.next, %latch9 ]
  %x9.next = add i64 %x9, %k
  %ov9 = icmp ult i64 %x9.next, %x9
  br i1 %ov9, label %trap, label %latch9
latch9:
  %i9.next = add nuw nsw i64 %i9, 1
  %c9 = icmp ne i64 %i9.next, %n9
  br i1 %c9, label %loop9, label %pre10
pre10:
  %r10 = urem i64 %x9.next, 6
  %n10 = add nuw nsw i64 %r10, 1
  br label %loop10
loop10:
  %x10 = phi i64 [ %x9.next, %pre10 ], [ %x10.next, %latch10 ]
  %i10 = phi i64 [ 0, %pre10 ], [ %i10.next, %latch10 ]
  %x10.next = add i64 %x10, %k
  %ov10 = icmp ult i64 %x10.next, %x10
  br i1 %ov10, label %trap, label %latch10
latch10:
  %i10.next = add nuw nsw i64 %i10, 1
  %c10 = icmp ne i64 %i10.next, %n10
  br i1 %c10, label %loop10, label %pre11
pre11:
  %r11 = urem i64 %x10.next, 6
  %n11 = add nuw nsw i64 %r11, 1
  br label %loop11
loop11:
  %x11 = phi i64 [ %x10.next, %pre11 ], [ %x11.next, %latch11 ]
  %i11 = phi i64 [ 0, %pre11 ], [ %i11.next, %latch11 ]
  %x11.next = add i64 %x11, %k
  %ov11 = icmp ult i64 %x11.next, %x11
  br i1 %ov11, label %trap, label %latch11
latch11:
  %i11.next = add nuw nsw i64 %i11, 1
  %c11 = icmp ne i64 %i11.next, %n11
  br i1 %c11, label %loop11, label %pre12
pre12:
  %r12 = urem i64 %x11.next, 6
  %n12 = add nuw nsw i64 %r12, 1
  br label %loop12
loop12:
  %x12 = phi i64 [ %x11.next, %pre12 ], [ %x12.next, %latch12 ]
  %i12 = phi i64 [ 0, %pre12 ], [ %i12.next, %latch12 ]
  %x12.next = add i64 %x12, %k
  %ov12 = icmp ult i64 %x12.next, %x12
  br i1 %ov12, label %trap, label %latch12
latch12:
  %i12.next = add nuw nsw i64 %i12, 1
  %c12 = icmp ne i64 %i12.next, %n12
  br i1 %c12, label %loop12, label %pre13
pre13:
  %r13 = urem i64 %x12.next, 6
  %n13 = add nuw nsw i64 %r13, 1
  br label %loop13
loop13:
  %x13 = phi i64 [ %x12.next, %pre13 ], [ %x13.next, %latch13 ]
  %i13 = phi i64 [ 0, %pre13 ], [ %i13.next, %latch13 ]
  %x13.next = add i64 %x13, %k
  %ov13 = icmp ult i64 %x13.next, %x13
  br i1 %ov13, label %trap, label %latch13
latch13:
  %i13.next = add nuw nsw i64 %i13, 1
  %c13 = icmp ne i64 %i13.next, %n13
  br i1 %c13, label %loop13, label %pre14
pre14:
  %r14 = urem i64 %x13.next, 6
  %n14 = add nuw nsw i64 %r14, 1
  br label %loop14
loop14:
  %x14 = phi i64 [ %x13.next, %pre14 ], [ %x14.next, %latch14 ]
  %i14 = phi i64 [ 0, %pre14 ], [ %i14.next, %latch14 ]
  %x14.next = add i64 %x14, %k
  %ov14 = icmp ult i64 %x14.next, %x14
  br i1 %ov14, label %trap, label %latch14
latch14:
  %i14.next = add nuw nsw i64 %i14, 1
  %c14 = icmp ne i64 %i14.next, %n14
  br i1 %c14, label %loop14, label %pre15
pre15:
  %r15 = urem i64 %x14.next, 6
  %n15 = add nuw nsw i64 %r15, 1
  br label %loop15
loop15:
  %x15 = phi i64 [ %x14.next, %pre15 ], [ %x15.next, %latch15 ]
  %i15 = phi i64 [ 0, %pre15 ], [ %i15.next, %latch15 ]
  %x15.next = add i64 %x15, %k
  %ov15 = icmp ult i64 %x15.next, %x15
  br i1 %ov15, label %trap, label %latch15
latch15:
  %i15.next = add nuw nsw i64 %i15, 1
  %c15 = icmp ne i64 %i15.next, %n15
  br i1 %c15, label %loop15, label %pre16
pre16:
  %r16 = urem i64 %x15.next, 6
  %n16 = add nuw nsw i64 %r16, 1
  br label %loop16
loop16:
  %x16 = phi i64 [ %x15.next, %pre16 ], [ %x16.next, %latch16 ]
  %i16 = phi i64 [ 0, %pre16 ], [ %i16.next, %latch16 ]
  %x16.next = add i64 %x16, %k
  %ov16 = icmp ult i64 %x16.next, %x16
  br i1 %ov16, label %trap, label %latch16
latch16:
  %i16.next = add nuw nsw i64 %i16, 1
  %c16 = icmp ne i64 %i16.next, %n16
  br i1 %c16, label %loop16, label %pre17
pre17:
  %r17 = urem i64 %x16.next, 6
  %n17 = add nuw nsw i64 %r17, 1
  br label %loop17
loop17:
  %x17 = phi i64 [ %x16.next, %pre17 ], [ %x17.next, %latch17 ]
  %i17 = phi i64 [ 0, %pre17 ], [ %i17.next, %latch17 ]
  %x17.next = add i64 %x17, %k
  %ov17 = icmp ult i64 %x17.next, %x17
  br i1 %ov17, label %trap, label %latch17
latch17:
  %i17.next = add nuw nsw i64 %i17, 1
  %c17 = icmp ne i64 %i17.next, %n17
  br i1 %c17, label %loop17, label %pre18
pre18:
  %r18 = urem i64 %x17.next, 6
  %n18 = add nuw nsw i64 %r18, 1
  br label %loop18
loop18:
  %x18 = phi i64 [ %x17.next, %pre18 ], [ %x18.next, %latch18 ]
  %i18 = phi i64 [ 0, %pre18 ], [ %i18.next, %latch18 ]
  %x18.next = add i64 %x18, %k
  %ov18 = icmp ult i64 %x18.next, %x18
  br i1 %ov18, label %trap, label %latch18
latch18:
  %i18.next = add nuw nsw i64 %i18, 1
  %c18 = icmp ne i64 %i18.next, %n18
  br i1 %c18, label %loop18, label %pre19
pre19:
  %r19 = urem i64 %x18.next, 6
  %n19 = add nuw nsw i64 %r19, 1
  br label %loop19
loop19:
  %x19 = phi i64 [ %x18.next, %pre19 ], [ %x19.next, %latch19 ]
  %i19 = phi i64 [ 0, %pre19 ], [ %i19.next, %latch19 ]
  %x19.next = add i64 %x19, %k
  %ov19 = icmp ult i64 %x19.next, %x19
  br i1 %ov19, label %trap, label %latch19
latch19:
  %i19.next = add nuw nsw i64 %i19, 1
  %c19 = icmp ne i64 %i19.next, %n19
  br i1 %c19, label %loop19, label %pre20
pre20:
  %r20 = urem i64 %x19.next, 6
  %n20 = add nuw nsw i64 %r20, 1
  br label %loop20
loop20:
  %x20 = phi i64 [ %x19.next, %pre20 ], [ %x20.next, %latch20 ]
  %i20 = phi i64 [ 0, %pre20 ], [ %i20.next, %latch20 ]
  %x20.next = add i64 %x20, %k
  %ov20 = icmp ult i64 %x20.next, %x20
  br i1 %ov20, label %trap, label %latch20
latch20:
  %i20.next = add nuw nsw i64 %i20, 1
  %c20 = icmp ne i64 %i20.next, %n20
  br i1 %c20, label %loop20, label %pre21
pre21:
  %r21 = urem i64 %x20.next, 6
  %n21 = add nuw nsw i64 %r21, 1
  br label %loop21
loop21:
  %x21 = phi i64 [ %x20.next, %pre21 ], [ %x21.next, %latch21 ]
  %i21 = phi i64 [ 0, %pre21 ], [ %i21.next, %latch21 ]
  %x21.next = add i64 %x21, %k
  %ov21 = icmp ult i64 %x21.next, %x21
  br i1 %ov21, label %trap, label %latch21
latch21:
  %i21.next = add nuw nsw i64 %i21, 1
  %c21 = icmp ne i64 %i21.next, %n21
  br i1 %c21, label %loop21, label %pre22
pre22:
  %r22 = urem i64 %x21.next, 6
  %n22 = add nuw nsw i64 %r22, 1
  br label %loop22
loop22:
  %x22 = phi i64 [ %x21.next, %pre22 ], [ %x22.next, %latch22 ]
  %i22 = phi i64 [ 0, %pre22 ], [ %i22.next, %latch22 ]
  %x22.next = add i64 %x22, %k
  %ov22 = icmp ult i64 %x22.next, %x22
  br i1 %ov22, label %trap, label %latch22
latch22:
  %i22.next = add nuw nsw i64 %i22, 1
  %c22 = icmp ne i64 %i22.next, %n22
  br i1 %c22, label %loop22, label %pre23
pre23:
  %r23 = urem i64 %x22.next, 6
  %n23 = add nuw nsw i64 %r23, 1
  br label %loop23
loop23:
  %x23 = phi i64 [ %x22.next, %pre23 ], [ %x23.next, %latch23 ]
  %i23 = phi i64 [ 0, %pre23 ], [ %i23.next, %latch23 ]
  %x23.next = add i64 %x23, %k
  %ov23 = icmp ult i64 %x23.next, %x23
  br i1 %ov23, label %trap, label %latch23
latch23:
  %i23.next = add nuw nsw i64 %i23, 1
  %c23 = icmp ne i64 %i23.next, %n23
  br i1 %c23, label %loop23, label %exit
exit:
  ret i64 %x23.next
trap:
  call void @llvm.trap()
  unreachable
}

declare void @llvm.trap()
