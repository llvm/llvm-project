; RUN: llc < %s -mtriple=arm64-eabi -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: ldp_int
; CHECK: ldp
define i32 @ldp_int(i32* %p) nounwind {
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %add = add nsw i32 %tmp1, %tmp
  ret i32 %add
}

; CHECK-LABEL: ldp_sext_int
; CHECK: ldpsw
define i64 @ldp_sext_int(i32* %p) nounwind {
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

; CHECK-LABEL: ldp_half_sext_res0_int:
; CHECK: ldp     w[[DST1:[0-9]+]], w[[DST2:[0-9]+]], [x0]
; CHECK: sxtw     x[[DST1]], w[[DST1]]
define i64 @ldp_half_sext_res0_int(i32* %p) nounwind {
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = zext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

; CHECK-LABEL: ldp_half_sext_res1_int:
; CHECK: ldp     w[[DST1:[0-9]+]], w[[DST2:[0-9]+]], [x0]
; CHECK: sxtw     x[[DST2]], w[[DST2]]
define i64 @ldp_half_sext_res1_int(i32* %p) nounwind {
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = zext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}


; CHECK-LABEL: ldp_long
; CHECK: ldp
define i64 @ldp_long(i64* %p) nounwind {
  %tmp = load i64, i64* %p, align 8
  %add.ptr = getelementptr inbounds i64, i64* %p, i64 1
  %tmp1 = load i64, i64* %add.ptr, align 8
  %add = add nsw i64 %tmp1, %tmp
  ret i64 %add
}

; CHECK-LABEL: ldp_float
; CHECK: ldp
define float @ldp_float(float* %p) nounwind {
  %tmp = load float, float* %p, align 4
  %add.ptr = getelementptr inbounds float, float* %p, i64 1
  %tmp1 = load float, float* %add.ptr, align 4
  %add = fadd float %tmp, %tmp1
  ret float %add
}

; CHECK-LABEL: ldp_double
; CHECK: ldp
define double @ldp_double(double* %p) nounwind {
  %tmp = load double, double* %p, align 8
  %add.ptr = getelementptr inbounds double, double* %p, i64 1
  %tmp1 = load double, double* %add.ptr, align 8
  %add = fadd double %tmp, %tmp1
  ret double %add
}

; CHECK-LABEL: ldp_doublex2
; CHECK: ldp
define <2 x double> @ldp_doublex2(<2 x double>* %p) nounwind {
  %tmp = load <2 x double>, <2 x double>* %p, align 16
  %add.ptr = getelementptr inbounds <2 x double>, <2 x double>* %p, i64 1
  %tmp1 = load <2 x double>, <2 x double>* %add.ptr, align 16
  %add = fadd <2 x double> %tmp, %tmp1
  ret <2 x double> %add
}

; Test the load/store optimizer---combine ldurs into a ldp, if appropriate
define i32 @ldur_int(i32* %a) nounwind {
; CHECK-LABEL: ldur_int
; CHECK: ldp     [[DST1:w[0-9]+]], [[DST2:w[0-9]+]], [x0, #-8]
; CHECK-NEXT: add     w{{[0-9]+}}, [[DST2]], [[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i32 -1
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 -2
  %tmp2 = load i32, i32* %p2, align 2
  %tmp3 = add i32 %tmp1, %tmp2
  ret i32 %tmp3
}

define i64 @ldur_sext_int(i32* %a) nounwind {
; CHECK-LABEL: ldur_sext_int
; CHECK: ldpsw     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-8]
; CHECK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i32 -1
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 -2
  %tmp2 = load i32, i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @ldur_half_sext_int_res0(i32* %a) nounwind {
; CHECK-LABEL: ldur_half_sext_int_res0
; CHECK: ldp     w[[DST1:[0-9]+]], w[[DST2:[0-9]+]], [x0, #-8]
; CHECK: sxtw     x[[DST1]], w[[DST1]]
; CHECK-NEXT: add     x{{[0-9]+}}, x[[DST2]], x[[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i32 -1
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 -2
  %tmp2 = load i32, i32* %p2, align 2
  %sexttmp1 = zext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @ldur_half_sext_int_res1(i32* %a) nounwind {
; CHECK-LABEL: ldur_half_sext_int_res1
; CHECK: ldp     w[[DST1:[0-9]+]], w[[DST2:[0-9]+]], [x0, #-8]
; CHECK: sxtw     x[[DST2]], w[[DST2]]
; CHECK-NEXT: add     x{{[0-9]+}}, x[[DST2]], x[[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i32 -1
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i32 -2
  %tmp2 = load i32, i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = zext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}


define i64 @ldur_long(i64* %a) nounwind ssp {
; CHECK-LABEL: ldur_long
; CHECK: ldp     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-16]
; CHECK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -1
  %tmp1 = load i64, i64* %p1, align 2
  %p2 = getelementptr inbounds i64, i64* %a, i64 -2
  %tmp2 = load i64, i64* %p2, align 2
  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define float @ldur_float(float* %a) {
; CHECK-LABEL: ldur_float
; CHECK: ldp     [[DST1:s[0-9]+]], [[DST2:s[0-9]+]], [x0, #-8]
; CHECK-NEXT: fadd    s{{[0-9]+}}, [[DST2]], [[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds float, float* %a, i64 -1
  %tmp1 = load float, float* %p1, align 2
  %p2 = getelementptr inbounds float, float* %a, i64 -2
  %tmp2 = load float, float* %p2, align 2
  %tmp3 = fadd float %tmp1, %tmp2
  ret float %tmp3
}

define double @ldur_double(double* %a) {
; CHECK-LABEL: ldur_double
; CHECK: ldp     [[DST1:d[0-9]+]], [[DST2:d[0-9]+]], [x0, #-16]
; CHECK-NEXT: fadd    d{{[0-9]+}}, [[DST2]], [[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds double, double* %a, i64 -1
  %tmp1 = load double, double* %p1, align 2
  %p2 = getelementptr inbounds double, double* %a, i64 -2
  %tmp2 = load double, double* %p2, align 2
  %tmp3 = fadd double %tmp1, %tmp2
  ret double %tmp3
}

define <2 x double> @ldur_doublex2(<2 x double>* %a) {
; CHECK-LABEL: ldur_doublex2
; CHECK: ldp     q[[DST1:[0-9]+]], q[[DST2:[0-9]+]], [x0, #-32]
; CHECK-NEXT: fadd    v{{[0-9]+}}.2d, v[[DST2]].2d, v[[DST1]].2d
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds <2 x double>, <2 x double>* %a, i64 -1
  %tmp1 = load <2 x double>, <2 x double>* %p1, align 2
  %p2 = getelementptr inbounds <2 x double>, <2 x double>* %a, i64 -2
  %tmp2 = load <2 x double>, <2 x double>* %p2, align 2
  %tmp3 = fadd <2 x double> %tmp1, %tmp2
  ret <2 x double> %tmp3
}

; Now check some boundary conditions
define i64 @pairUpBarelyIn(i64* %a) nounwind ssp {
; CHECK-LABEL: pairUpBarelyIn
; CHECK-NOT: ldur
; CHECK: ldp     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-256]
; CHECK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -31
  %tmp1 = load i64, i64* %p1, align 2
  %p2 = getelementptr inbounds i64, i64* %a, i64 -32
  %tmp2 = load i64, i64* %p2, align 2
  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyInSext(i32* %a) nounwind ssp {
; CHECK-LABEL: pairUpBarelyInSext
; CHECK-NOT: ldur
; CHECK: ldpsw     [[DST1:x[0-9]+]], [[DST2:x[0-9]+]], [x0, #-256]
; CHECK-NEXT: add     x{{[0-9]+}}, [[DST2]], [[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -63
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i64 -64
  %tmp2 = load i32, i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyInHalfSextRes0(i32* %a) nounwind ssp {
; CHECK-LABEL: pairUpBarelyInHalfSextRes0
; CHECK-NOT: ldur
; CHECK: ldp     w[[DST1:[0-9]+]], w[[DST2:[0-9]+]], [x0, #-256]
; CHECK: sxtw     x[[DST1]], w[[DST1]]
; CHECK-NEXT: add     x{{[0-9]+}}, x[[DST2]], x[[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -63
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i64 -64
  %tmp2 = load i32, i32* %p2, align 2
  %sexttmp1 = zext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyInHalfSextRes1(i32* %a) nounwind ssp {
; CHECK-LABEL: pairUpBarelyInHalfSextRes1
; CHECK-NOT: ldur
; CHECK: ldp     w[[DST1:[0-9]+]], w[[DST2:[0-9]+]], [x0, #-256]
; CHECK: sxtw     x[[DST2]], w[[DST2]]
; CHECK-NEXT: add     x{{[0-9]+}}, x[[DST2]], x[[DST1]]
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -63
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i64 -64
  %tmp2 = load i32, i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = zext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyOut(i64* %a) nounwind ssp {
; CHECK-LABEL: pairUpBarelyOut
; CHECK-NOT: ldp
; Don't be fragile about which loads or manipulations of the base register
; are used---just check that there isn't an ldp before the add
; CHECK: add
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -32
  %tmp1 = load i64, i64* %p1, align 2
  %p2 = getelementptr inbounds i64, i64* %a, i64 -33
  %tmp2 = load i64, i64* %p2, align 2
  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @pairUpBarelyOutSext(i32* %a) nounwind ssp {
; CHECK-LABEL: pairUpBarelyOutSext
; CHECK-NOT: ldp
; Don't be fragile about which loads or manipulations of the base register
; are used---just check that there isn't an ldp before the add
; CHECK: add
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -64
  %tmp1 = load i32, i32* %p1, align 2
  %p2 = getelementptr inbounds i32, i32* %a, i64 -65
  %tmp2 = load i32, i32* %p2, align 2
  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
  ret i64 %tmp3
}

define i64 @pairUpNotAligned(i64* %a) nounwind ssp {
; CHECK-LABEL: pairUpNotAligned
; CHECK-NOT: ldp
; CHECK: ldur
; CHECK-NEXT: ldur
; CHECK-NEXT: add
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i64, i64* %a, i64 -18
  %bp1 = bitcast i64* %p1 to i8*
  %bp1p1 = getelementptr inbounds i8, i8* %bp1, i64 1
  %dp1 = bitcast i8* %bp1p1 to i64*
  %tmp1 = load i64, i64* %dp1, align 1

  %p2 = getelementptr inbounds i64, i64* %a, i64 -17
  %bp2 = bitcast i64* %p2 to i8*
  %bp2p1 = getelementptr inbounds i8, i8* %bp2, i64 1
  %dp2 = bitcast i8* %bp2p1 to i64*
  %tmp2 = load i64, i64* %dp2, align 1

  %tmp3 = add i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @pairUpNotAlignedSext(i32* %a) nounwind ssp {
; CHECK-LABEL: pairUpNotAlignedSext
; CHECK-NOT: ldp
; CHECK: ldursw
; CHECK-NEXT: ldursw
; CHECK-NEXT: add
; CHECK-NEXT: ret
  %p1 = getelementptr inbounds i32, i32* %a, i64 -18
  %bp1 = bitcast i32* %p1 to i8*
  %bp1p1 = getelementptr inbounds i8, i8* %bp1, i64 1
  %dp1 = bitcast i8* %bp1p1 to i32*
  %tmp1 = load i32, i32* %dp1, align 1

  %p2 = getelementptr inbounds i32, i32* %a, i64 -17
  %bp2 = bitcast i32* %p2 to i8*
  %bp2p1 = getelementptr inbounds i8, i8* %bp2, i64 1
  %dp2 = bitcast i8* %bp2p1 to i32*
  %tmp2 = load i32, i32* %dp2, align 1

  %sexttmp1 = sext i32 %tmp1 to i64
  %sexttmp2 = sext i32 %tmp2 to i64
  %tmp3 = add i64 %sexttmp1, %sexttmp2
 ret i64 %tmp3
}

declare void @use-ptr(i32*)

; CHECK-LABEL: ldp_sext_int_pre
; CHECK: ldpsw x{{[0-9]+}}, x{{[0-9]+}}, [x{{[0-9]+}}, #8]
define i64 @ldp_sext_int_pre(i32* %p) nounwind {
  %ptr = getelementptr inbounds i32, i32* %p, i64 2
  call void @use-ptr(i32* %ptr)
  %add.ptr = getelementptr inbounds i32, i32* %ptr, i64 0
  %tmp = load i32, i32* %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i32, i32* %ptr, i64 1
  %tmp1 = load i32, i32* %add.ptr1, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

; CHECK-LABEL: ldp_sext_int_post
; CHECK: ldpsw x{{[0-9]+}}, x{{[0-9]+}}, [x0], #8
define i64 @ldp_sext_int_post(i32* %p) nounwind {
  %tmp = load i32, i32* %p, align 4
  %add.ptr = getelementptr inbounds i32, i32* %p, i64 1
  %tmp1 = load i32, i32* %add.ptr, align 4
  %sexttmp = sext i32 %tmp to i64
  %sexttmp1 = sext i32 %tmp1 to i64
  %ptr = getelementptr inbounds i32, i32* %add.ptr, i64 1
  call void @use-ptr(i32* %ptr)
  %add = add nsw i64 %sexttmp1, %sexttmp
  ret i64 %add
}

