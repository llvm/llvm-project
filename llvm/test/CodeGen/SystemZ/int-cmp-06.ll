; Test 64-bit comparison in which the second operand is a zero-extended i32.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare i64 @foo()

; Check unsigned register comparison.
define double @f1(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK-LABEL: f1:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation.
define double @f2(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK-LABEL: f2:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed register comparison, which can't use CLGFR.
define double @f3(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK-LABEL: f3:
; CHECK-NOT: clgfr
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation
define double @f4(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK-LABEL: f4:
; CHECK-NOT: clgfr
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check register equality.
define double @f5(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK-LABEL: f5:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation
define double @f6(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK-LABEL: f6:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check register inequality.
define double @f7(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK-LABEL: f7:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: blhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with a different representation
define double @f8(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK-LABEL: f8:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: blhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check unsigned comparison with memory.
define double @f9(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK-LABEL: f9:
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check signed comparison with memory.
define double @f10(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK-LABEL: f10:
; CHECK-NOT: clgf
; CHECK: br %r14
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp slt i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check memory equality.
define double @f11(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK-LABEL: f11:
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: ber %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp eq i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check memory inequality.
define double @f12(double %a, double %b, i64 %i1, i32 *%ptr) {
; CHECK-LABEL: f12:
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: blhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ne i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the aligned CLGF range.
define double @f13(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK-LABEL: f13:
; CHECK: clgf %r2, 524284(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 131071
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f14(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK-LABEL: f14:
; CHECK: agfi %r3, 524288
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 131072
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the high end of the negative aligned CLGF range.
define double @f15(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK-LABEL: f15:
; CHECK: clgf %r2, -4(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -1
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the low end of the CLGF range.
define double @f16(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK-LABEL: f16:
; CHECK: clgf %r2, -524288(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -131072
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the next word down, which needs separate address logic.
; Other sequences besides this one would be OK.
define double @f17(double %a, double %b, i64 %i1, i32 *%base) {
; CHECK-LABEL: f17:
; CHECK: agfi %r3, -524292
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%base, i64 -131073
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that CLGF allows an index.
define double @f18(double %a, double %b, i64 %i1, i64 %base, i64 %index) {
; CHECK-LABEL: f18:
; CHECK: clgf %r2, 524284({{%r4,%r3|%r3,%r4}})
; CHECK-NEXT: blr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %add1 = add i64 %base, %index
  %add2 = add i64 %add1, 524284
  %ptr = inttoptr i64 %add2 to i32 *
  %unext = load i32, i32 *%ptr
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check that comparisons of spilled values can use CLGF rather than CLGFR.
define i64 @f19(i32 *%ptr0) {
; CHECK-LABEL: f19:
; CHECK: brasl %r14, foo@PLT
; CHECK: clgf {{%r[0-9]+}}, 16{{[04]}}(%r15)
; CHECK: br %r14
  %ptr1 = getelementptr i32, i32 *%ptr0, i64 2
  %ptr2 = getelementptr i32, i32 *%ptr0, i64 4
  %ptr3 = getelementptr i32, i32 *%ptr0, i64 6
  %ptr4 = getelementptr i32, i32 *%ptr0, i64 8
  %ptr5 = getelementptr i32, i32 *%ptr0, i64 10
  %ptr6 = getelementptr i32, i32 *%ptr0, i64 12
  %ptr7 = getelementptr i32, i32 *%ptr0, i64 14
  %ptr8 = getelementptr i32, i32 *%ptr0, i64 16
  %ptr9 = getelementptr i32, i32 *%ptr0, i64 18

  %val0 = load i32, i32 *%ptr0
  %val1 = load i32, i32 *%ptr1
  %val2 = load i32, i32 *%ptr2
  %val3 = load i32, i32 *%ptr3
  %val4 = load i32, i32 *%ptr4
  %val5 = load i32, i32 *%ptr5
  %val6 = load i32, i32 *%ptr6
  %val7 = load i32, i32 *%ptr7
  %val8 = load i32, i32 *%ptr8
  %val9 = load i32, i32 *%ptr9

  %frob0 = add i32 %val0, 100
  %frob1 = add i32 %val1, 100
  %frob2 = add i32 %val2, 100
  %frob3 = add i32 %val3, 100
  %frob4 = add i32 %val4, 100
  %frob5 = add i32 %val5, 100
  %frob6 = add i32 %val6, 100
  %frob7 = add i32 %val7, 100
  %frob8 = add i32 %val8, 100
  %frob9 = add i32 %val9, 100

  store i32 %frob0, i32 *%ptr0
  store i32 %frob1, i32 *%ptr1
  store i32 %frob2, i32 *%ptr2
  store i32 %frob3, i32 *%ptr3
  store i32 %frob4, i32 *%ptr4
  store i32 %frob5, i32 *%ptr5
  store i32 %frob6, i32 *%ptr6
  store i32 %frob7, i32 *%ptr7
  store i32 %frob8, i32 *%ptr8
  store i32 %frob9, i32 *%ptr9

  %ret = call i64 @foo()

  %ext0 = zext i32 %frob0 to i64
  %ext1 = zext i32 %frob1 to i64
  %ext2 = zext i32 %frob2 to i64
  %ext3 = zext i32 %frob3 to i64
  %ext4 = zext i32 %frob4 to i64
  %ext5 = zext i32 %frob5 to i64
  %ext6 = zext i32 %frob6 to i64
  %ext7 = zext i32 %frob7 to i64
  %ext8 = zext i32 %frob8 to i64
  %ext9 = zext i32 %frob9 to i64

  %cmp0 = icmp ult i64 %ret, %ext0
  %cmp1 = icmp ult i64 %ret, %ext1
  %cmp2 = icmp ult i64 %ret, %ext2
  %cmp3 = icmp ult i64 %ret, %ext3
  %cmp4 = icmp ult i64 %ret, %ext4
  %cmp5 = icmp ult i64 %ret, %ext5
  %cmp6 = icmp ult i64 %ret, %ext6
  %cmp7 = icmp ult i64 %ret, %ext7
  %cmp8 = icmp ult i64 %ret, %ext8
  %cmp9 = icmp ult i64 %ret, %ext9

  %sel0 = select i1 %cmp0, i64 %ret, i64 0
  %sel1 = select i1 %cmp1, i64 %sel0, i64 1
  %sel2 = select i1 %cmp2, i64 %sel1, i64 2
  %sel3 = select i1 %cmp3, i64 %sel2, i64 3
  %sel4 = select i1 %cmp4, i64 %sel3, i64 4
  %sel5 = select i1 %cmp5, i64 %sel4, i64 5
  %sel6 = select i1 %cmp6, i64 %sel5, i64 6
  %sel7 = select i1 %cmp7, i64 %sel6, i64 7
  %sel8 = select i1 %cmp8, i64 %sel7, i64 8
  %sel9 = select i1 %cmp9, i64 %sel8, i64 9

  ret i64 %sel9
}

; Check the comparison can be reversed if that allows CLGFR to be used.
define double @f20(double %a, double %b, i64 %i1, i32 %unext) {
; CHECK-LABEL: f20:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = zext i32 %unext to i64
  %cond = icmp ult i64 %i2, %i1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; ...and again with the AND representation.
define double @f21(double %a, double %b, i64 %i1, i64 %unext) {
; CHECK-LABEL: f21:
; CHECK: clgfr %r2, %r3
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %i2 = and i64 %unext, 4294967295
  %cond = icmp ult i64 %i2, %i1
  %res = select i1 %cond, double %a, double %b
  ret double %res
}

; Check the comparison can be reversed if that allows CLGF to be used.
define double @f22(double %a, double %b, i64 %i2, i32 *%ptr) {
; CHECK-LABEL: f22:
; CHECK: clgf %r2, 0(%r3)
; CHECK-NEXT: bhr %r14
; CHECK: ldr %f0, %f2
; CHECK: br %r14
  %unext = load i32, i32 *%ptr
  %i1 = zext i32 %unext to i64
  %cond = icmp ult i64 %i1, %i2
  %res = select i1 %cond, double %a, double %b
  ret double %res
}
