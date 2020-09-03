; RUN: opt -S -analyze -scalar-evolution < %s | FileCheck %s

!0 = !{i8 0, i8 127}

define void @f0(i8* %len_addr) {
; CHECK-LABEL: Classifying expressions for: @f0
 entry:
  %len = load i8, i8* %len_addr, !range !0
  %len_norange = load i8, i8* %len_addr
; CHECK:  %len = load i8, i8* %len_addr, align 1, !range !0
; CHECK-NEXT:  -->  %len U: [0,127) S: [0,127)
; CHECK:  %len_norange = load i8, i8* %len_addr
; CHECK-NEXT:  -->  %len_norange U: full-set S: full-set

  %t0 = add i8 %len, 1
  %t1 = add i8 %len, 2
; CHECK:  %t0 = add i8 %len, 1
; CHECK-NEXT:  -->  (1 + %len)<nuw><nsw> U: [1,-128) S: [1,-128)
; CHECK:  %t1 = add i8 %len, 2
; CHECK-NEXT:  -->  (2 + %len)<nuw> U: [2,-127) S: [2,-127)

  %t2 = sub i8 %len, 1
  %t3 = sub i8 %len, 2
; CHECK:  %t2 = sub i8 %len, 1
; CHECK-NEXT:  -->  (-1 + %len)<nsw> U: [-1,126) S: [-1,126)
; CHECK:  %t3 = sub i8 %len, 2
; CHECK-NEXT:  -->  (-2 + %len)<nsw> U: [-2,125) S: [-2,125)

  %q0 = add i8 %len_norange, 1
  %q1 = add i8 %len_norange, 2
; CHECK:  %q0 = add i8 %len_norange, 1
; CHECK-NEXT:  -->  (1 + %len_norange) U: full-set S: full-set
; CHECK:  %q1 = add i8 %len_norange, 2
; CHECK-NEXT:  -->  (2 + %len_norange) U: full-set S: full-set

  %q2 = sub i8 %len_norange, 1
  %q3 = sub i8 %len_norange, 2
; CHECK:  %q2 = sub i8 %len_norange, 1
; CHECK-NEXT:  -->  (-1 + %len_norange) U: full-set S: full-set
; CHECK:  %q3 = sub i8 %len_norange, 2
; CHECK-NEXT:  -->  (-2 + %len_norange) U: full-set S: full-set

  ret void
}

define void @f1(i8* %len_addr) {
; CHECK-LABEL: Classifying expressions for: @f1
 entry:
  %len = load i8, i8* %len_addr, !range !0
  %len_norange = load i8, i8* %len_addr
; CHECK:  %len = load i8, i8* %len_addr, align 1, !range !0
; CHECK-NEXT:  -->  %len U: [0,127) S: [0,127)
; CHECK:  %len_norange = load i8, i8* %len_addr
; CHECK-NEXT:  -->  %len_norange U: full-set S: full-set

  %t0 = add i8 %len, -1
  %t1 = add i8 %len, -2
; CHECK:  %t0 = add i8 %len, -1
; CHECK-NEXT:  -->  (-1 + %len)<nsw> U: [-1,126) S: [-1,126)
; CHECK:  %t1 = add i8 %len, -2
; CHECK-NEXT:  -->  (-2 + %len)<nsw> U: [-2,125) S: [-2,125)

  %t0.sext = sext i8 %t0 to i16
  %t1.sext = sext i8 %t1 to i16
; CHECK:  %t0.sext = sext i8 %t0 to i16
; CHECK-NEXT:  -->  (-1 + (zext i8 %len to i16))<nsw> U: [-1,126) S: [-1,126)
; CHECK:  %t1.sext = sext i8 %t1 to i16
; CHECK-NEXT:  -->  (-2 + (zext i8 %len to i16))<nsw> U: [-2,125) S: [-2,125)

  %q0 = add i8 %len_norange, 1
  %q1 = add i8 %len_norange, 2
; CHECK:  %q0 = add i8 %len_norange, 1
; CHECK-NEXT:  -->  (1 + %len_norange) U: full-set S: full-set
; CHECK:  %q1 = add i8 %len_norange, 2
; CHECK-NEXT:  -->  (2 + %len_norange) U: full-set S: full-set

  %q0.sext = sext i8 %q0 to i16
  %q1.sext = sext i8 %q1 to i16
; CHECK:  %q0.sext = sext i8 %q0 to i16
; CHECK-NEXT:  -->  (sext i8 (1 + %len_norange) to i16) U: [-128,128) S: [-128,128)
; CHECK:  %q1.sext = sext i8 %q1 to i16
; CHECK-NEXT:  -->  (sext i8 (2 + %len_norange) to i16) U: [-128,128) S: [-128,128)

  ret void
}

define void @f2(i8* %len_addr) {
; CHECK-LABEL: Classifying expressions for: @f2
 entry:
  %len = load i8, i8* %len_addr, !range !0
  %len_norange = load i8, i8* %len_addr
; CHECK:  %len = load i8, i8* %len_addr, align 1, !range !0
; CHECK-NEXT:  -->  %len U: [0,127) S: [0,127)
; CHECK:  %len_norange = load i8, i8* %len_addr
; CHECK-NEXT:  -->  %len_norange U: full-set S: full-set

  %t0 = add i8 %len, 1
  %t1 = add i8 %len, 2
; CHECK:  %t0 = add i8 %len, 1
; CHECK-NEXT:  -->  (1 + %len)<nuw><nsw>
; CHECK:  %t1 = add i8 %len, 2
; CHECK-NEXT:  -->  (2 + %len)<nuw>

  %t0.zext = zext i8 %t0 to i16
  %t1.zext = zext i8 %t1 to i16
; CHECK:  %t0.zext = zext i8 %t0 to i16
; CHECK-NEXT: -->  (1 + (zext i8 %len to i16))<nuw><nsw> U: [1,128) S: [1,128)
; CHECK:  %t1.zext = zext i8 %t1 to i16
; CHECK-NEXT:  -->  (2 + (zext i8 %len to i16))<nuw><nsw> U: [2,129) S: [2,129)

  %q0 = add i8 %len_norange, 1
  %q1 = add i8 %len_norange, 2
  %q0.zext = zext i8 %q0 to i16
  %q1.zext = zext i8 %q1 to i16

; CHECK:  %q0.zext = zext i8 %q0 to i16
; CHECK-NEXT:  -->  (zext i8 (1 + %len_norange) to i16) U: [0,256) S: [0,256)
; CHECK:  %q1.zext = zext i8 %q1 to i16
; CHECK-NEXT:  -->  (zext i8 (2 + %len_norange) to i16) U: [0,256) S: [0,256)

  ret void
}

@z_addr = external global [16 x i8], align 4
@z_addr_noalign = external global [16 x i8]

%union = type { [10 x [4 x float]] }
@tmp_addr = external unnamed_addr global { %union, [2000 x i8] }

define void @f3(i8* %x_addr, i8* %y_addr, i32* %tmp_addr) {
; CHECK-LABEL: Classifying expressions for: @f3
 entry:
  %x = load i8, i8* %x_addr
  %t0 = mul i8 %x, 4
  %t1 = add i8 %t0, 5
  %t1.zext = zext i8 %t1 to i16
; CHECK:  %t1.zext = zext i8 %t1 to i16
; CHECK-NEXT:  -->  (1 + (zext i8 (4 + (4 * %x)) to i16))<nuw><nsw> U: [1,254) S: [1,257)

  %q0 = mul i8 %x, 4
  %q1 = add i8 %q0, 7
  %q1.zext = zext i8 %q1 to i16
; CHECK:  %q1.zext = zext i8 %q1 to i16
; CHECK-NEXT:  -->  (3 + (zext i8 (4 + (4 * %x)) to i16))<nuw><nsw> U: [3,256) S: [3,259)

  %p0 = mul i8 %x, 4
  %p1 = add i8 %p0, 8
  %p1.zext = zext i8 %p1 to i16
; CHECK:  %p1.zext = zext i8 %p1 to i16
; CHECK-NEXT:  -->  (zext i8 (8 + (4 * %x)) to i16) U: [0,253) S: [0,256)

  %r0 = mul i8 %x, 4
  %r1 = add i8 %r0, 254
  %r1.zext = zext i8 %r1 to i16
; CHECK:  %r1.zext = zext i8 %r1 to i16
; CHECK-NEXT:  -->  (2 + (zext i8 (-4 + (4 * %x)) to i16))<nuw><nsw> U: [2,255) S: [2,258)

  %y = load i8, i8* %y_addr
  %s0 = mul i8 %x, 32
  %s1 = mul i8 %y, 36
  %s2 = add i8 %s0, %s1
  %s3 = add i8 %s2, 5
  %s3.zext = zext i8 %s3 to i16
; CHECK:  %s3.zext = zext i8 %s3 to i16
; CHECK-NEXT:  -->  (1 + (zext i8 (4 + (32 * %x) + (36 * %y)) to i16))<nuw><nsw> U: [1,254) S: [1,257)

  %ptr = bitcast [16 x i8]* @z_addr to i8*
  %int0 = ptrtoint i8* %ptr to i32
  %int5 = add i32 %int0, 5
  %int.zext = zext i32 %int5 to i64
; CHECK:  %int.zext = zext i32 %int5 to i64
; CHECK-NEXT:  -->  (1 + (zext i32 (4 + %int0) to i64))<nuw><nsw> U: [1,4294967294) S: [1,4294967297)

  %ptr_noalign = bitcast [16 x i8]* @z_addr_noalign to i8*
  %int0_na = ptrtoint i8* %ptr_noalign to i32
  %int5_na = add i32 %int0_na, 5
  %int.zext_na = zext i32 %int5_na to i64
; CHECK:  %int.zext_na = zext i32 %int5_na to i64
; CHECK-NEXT:  -->  (zext i32 (5 + %int0_na) to i64) U: [0,4294967296) S: [0,4294967296)

  %tmp = load i32, i32* %tmp_addr
  %mul = and i32 %tmp, -4
  %add4 = add i32 %mul, 4
  %add4.zext = zext i32 %add4 to i64
  %sunkaddr3 = mul i64 %add4.zext, 4
  %sunkaddr4 = getelementptr inbounds i8, i8* bitcast ({ %union, [2000 x i8] }* @tmp_addr to i8*), i64 %sunkaddr3
  %sunkaddr5 = getelementptr inbounds i8, i8* %sunkaddr4, i64 4096
  %addr4.cast = bitcast i8* %sunkaddr5 to i32*
  %addr4.incr = getelementptr i32, i32* %addr4.cast, i64 1
; CHECK:  %addr4.incr = getelementptr i32, i32* %addr4.cast, i64 1
; CHECK-NEXT:  -->  ([[C:4100]] + ([[SIZE:4]] * (zext i32 ([[OFFSET:4]] + ([[STRIDE:4]] * (%tmp /u [[STRIDE]]))<nuw>) to i64))<nuw><nsw> + @tmp_addr)

  %add5 = add i32 %mul, 5
  %add5.zext = zext i32 %add5 to i64
  %sunkaddr0 = mul i64 %add5.zext, 4
  %sunkaddr1 = getelementptr inbounds i8, i8* bitcast ({ %union, [2000 x i8] }* @tmp_addr to i8*), i64 %sunkaddr0
  %sunkaddr2 = getelementptr inbounds i8, i8* %sunkaddr1, i64 4096
  %addr5.cast = bitcast i8* %sunkaddr2 to i32*
; CHECK:  %addr5.cast = bitcast i8* %sunkaddr2 to i32*
; CHECK-NEXT:  -->  ([[C]]    +   ([[SIZE]]  *  (zext i32 ([[OFFSET]]  +  ([[STRIDE]]  *  (%tmp /u [[STRIDE]]))<nuw>) to i64))<nuw><nsw> + @tmp_addr)

  ret void
}
