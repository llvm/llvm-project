; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -ppc-vsr-nums-as-vr \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define fp128 @loadConstant() {
; CHECK-LABEL: loadConstant:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis [[REG0:[0-9]+]], 2, .LCPI0_0@toc@ha
; CHECK-NEXT:    addi [[REG0]], [[REG0]], .LCPI0_0@toc@l
; CHECK-NEXT:    lxvx 2, 0, [[REG0]]
; CHECK-NEXT:    blr
  entry:
    ret fp128 0xL00000000000000004001400000000000
}

; Function Attrs: norecurse nounwind readnone
define fp128 @loadConstant2(fp128 %a, fp128 %b) {
; CHECK-LABEL: loadConstant2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsaddqp 2, 2, 3
; CHECK-NEXT:    addis [[REG0:[0-9]+]], 2, .LCPI1_0@toc@ha
; CHECK-NEXT:    addi [[REG0]], [[REG0]], .LCPI1_0@toc@l
; CHECK-NEXT:    lxvx [[REG0]], 0, [[REG0]]
; CHECK-NEXT:    xsaddqp 2, 2, [[REG0]]
; CHECK-NEXT:    blr
  entry:
    %add = fadd fp128 %a, %b
      %add1 = fadd fp128 %add, 0xL00000000000000004001400000000000
        ret fp128 %add1
}

; Test passing float128 by value.
; Function Attrs: norecurse nounwind readnone
define signext i32 @fp128Param(fp128 %a) {
; CHECK-LABEL: fp128Param:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xscvqpswz 2, 2
; CHECK-NEXT:    mfvsrwz 3, 2
; CHECK-NEXT:    extsw 3, 3
; CHECK-NEXT:    blr
entry:
  %conv = fptosi fp128 %a to i32
  ret i32 %conv
}

; Test float128 as return value.
; Function Attrs: norecurse nounwind readnone
define fp128 @fp128Return(fp128 %a, fp128 %b) {
; CHECK-LABEL: fp128Return:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsaddqp 2, 2, 3
; CHECK-NEXT:    blr
entry:
  %add = fadd fp128 %a, %b
  ret fp128 %add
}

; array of float128 types
; Function Attrs: norecurse nounwind readonly
define fp128 @fp128Array(fp128* nocapture readonly %farray,
; CHECK-LABEL: fp128Array:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    sldi 4, 4, 4
; CHECK-NEXT:    lxv 2, 0(3)
; CHECK-NEXT:    add 4, 3, 4
; CHECK-NEXT:    lxv 3, -16(4)
; CHECK-NEXT:    xsaddqp 2, 2, 3
; CHECK-NEXT:    blr
                         i32 signext %loopcnt, fp128* nocapture readnone %sum) {
entry:
  %0 = load fp128, fp128* %farray, align 16
  %sub = add nsw i32 %loopcnt, -1
  %idxprom = sext i32 %sub to i64
  %arrayidx1 = getelementptr inbounds fp128, fp128* %farray, i64 %idxprom
  %1 = load fp128, fp128* %arrayidx1, align 16
  %add = fadd fp128 %0, %1
  ret fp128 %add
}

; Up to 12 qualified floating-point arguments can be passed in v2-v13.
; Function to test passing 13 float128 parameters.
; Function Attrs: norecurse nounwind readnone
define fp128 @maxVecParam(fp128 %p1, fp128 %p2, fp128 %p3, fp128 %p4, fp128 %p5,
; CHECK-LABEL: maxVecParam:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsaddqp 2, 2, 3
; CHECK-NEXT:    lxv [[REG0:[0-9]+]], 224(1)
; CHECK-NEXT:    xsaddqp 2, 2, 4
; CHECK-NEXT:    xsaddqp 2, 2, 5
; CHECK-NEXT:    xsaddqp 2, 2, 6
; CHECK-NEXT:    xsaddqp 2, 2, 7
; CHECK-NEXT:    xsaddqp 2, 2, 8
; CHECK-NEXT:    xsaddqp 2, 2, 9
; CHECK-NEXT:    xsaddqp 2, 2, 10
; CHECK-NEXT:    xsaddqp 2, 2, 11
; CHECK-NEXT:    xsaddqp 2, 2, 12
; CHECK-NEXT:    xsaddqp 2, 2, 13
; CHECK-NEXT:    xssubqp 2, 2, [[REG0]]
; CHECK-NEXT:    blr
                          fp128 %p6, fp128 %p7, fp128 %p8, fp128 %p9, fp128 %p10,
                          fp128 %p11, fp128 %p12, fp128 %p13) {
entry:
  %add = fadd fp128 %p1, %p2
  %add1 = fadd fp128 %add, %p3
  %add2 = fadd fp128 %add1, %p4
  %add3 = fadd fp128 %add2, %p5
  %add4 = fadd fp128 %add3, %p6
  %add5 = fadd fp128 %add4, %p7
  %add6 = fadd fp128 %add5, %p8
  %add7 = fadd fp128 %add6, %p9
  %add8 = fadd fp128 %add7, %p10
  %add9 = fadd fp128 %add8, %p11
  %add10 = fadd fp128 %add9, %p12
  %sub = fsub fp128 %add10, %p13
  ret fp128 %sub
}

; Passing a mix of float128 and other type parameters.
; Function Attrs: norecurse nounwind readnone
define fp128 @mixParam_01(fp128 %a, i32 signext %i, fp128 %b) {
; CHECK-LABEL: mixParam_01:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrwa 4, 5
; CHECK-NEXT:    xsaddqp 2, 2, 3
; CHECK-NEXT:    xscvsdqp [[REG0:[0-9]+]], 4
; CHECK-NEXT:    xsaddqp 2, 2, [[REG0]]
; CHECK-NEXT:    blr
entry:
  %add = fadd fp128 %a, %b
  %conv = sitofp i32 %i to fp128
  %add1 = fadd fp128 %add, %conv
  ret fp128 %add1
}
; Function Attrs: norecurse nounwind readnone
define fastcc fp128 @mixParam_01f(fp128 %a, i32 signext %i, fp128 %b) {
; CHECK-LABEL: mixParam_01f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrwa [[REG0:[0-9]+]], 3
; CHECK-NEXT:    xsaddqp 2, 2, 3
; CHECK-NEXT:    xscvsdqp [[REG1:[0-9]+]], [[REG0]]
; CHECK-NEXT:    xsaddqp 2, 2, [[REG1]]
; CHECK-NEXT:    blr
entry:
  %add = fadd fp128 %a, %b
  %conv = sitofp i32 %i to fp128
  %add1 = fadd fp128 %add, %conv
  ret fp128 %add1
}

; Function Attrs: norecurse nounwind
define fp128 @mixParam_02(fp128 %p1, double %p2, i64* nocapture %p3,
; CHECK-LABEL: mixParam_02:
; CHECK:       # %bb.0: # %entry
; CHECK-DAG:     lwz 3, 96(1)
; CHECK:         add 4, 7, 9
; CHECK-NEXT:    xxlor [[REG0:[0-9]+]], 1, 1
; CHECK-DAG:     add 4, 4, 10
; CHECK:         xscvdpqp [[REG0]], [[REG0]]
; CHECK-NEXT:    add 3, 4, 3
; CHECK-NEXT:    clrldi 3, 3, 32
; CHECK-NEXT:    std 3, 0(6)
; CHECK-NEXT:    lxv [[REG1:[0-9]+]], 0(8)
; CHECK-NEXT:    xsaddqp 2, [[REG1]], 2
; CHECK-NEXT:    xsaddqp 2, 2, 3
; CHECK-NEXT:    blr
                          i16 signext %p4, fp128* nocapture readonly %p5,
                          i32 signext %p6, i8 zeroext %p7, i32 zeroext %p8) {
entry:
  %conv = sext i16 %p4 to i32
  %add = add nsw i32 %conv, %p6
  %conv1 = zext i8 %p7 to i32
  %add2 = add nsw i32 %add, %conv1
  %add3 = add i32 %add2, %p8
  %conv4 = zext i32 %add3 to i64
  store i64 %conv4, i64* %p3, align 8
  %0 = load fp128, fp128* %p5, align 16
  %add5 = fadd fp128 %0, %p1
  %conv6 = fpext double %p2 to fp128
  %add7 = fadd fp128 %add5, %conv6
  ret fp128 %add7
}

; Function Attrs: norecurse nounwind
define fastcc fp128 @mixParam_02f(fp128 %p1, double %p2, i64* nocapture %p3,
; CHECK-LABEL: mixParam_02f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    add 4, 4, 6
; CHECK-NEXT:    xxlor [[REG0:[0-9]+]], 1, 1
; CHECK-NEXT:    add 4, 4, 7
; CHECK-NEXT:    xscvdpqp [[REG0]], [[REG0]]
; CHECK-NEXT:    add 4, 4, 8
; CHECK-NEXT:    clrldi 4, 4, 32
; CHECK-NEXT:    std 4, 0(3)
; CHECK-NEXT:    lxv [[REG1:[0-9]+]], 0(5)
; CHECK-NEXT:    xsaddqp 2, [[REG1]], 2
; CHECK-NEXT:    xsaddqp 2, 2, [[REG0]] 
; CHECK-NEXT:    blr
                                  i16 signext %p4, fp128* nocapture readonly %p5,
                                  i32 signext %p6, i8 zeroext %p7, i32 zeroext %p8) {
entry:
  %conv = sext i16 %p4 to i32
  %add = add nsw i32 %conv, %p6
  %conv1 = zext i8 %p7 to i32
  %add2 = add nsw i32 %add, %conv1
  %add3 = add i32 %add2, %p8
  %conv4 = zext i32 %add3 to i64
  store i64 %conv4, i64* %p3, align 8
  %0 = load fp128, fp128* %p5, align 16
  %add5 = fadd fp128 %0, %p1
  %conv6 = fpext double %p2 to fp128
  %add7 = fadd fp128 %add5, %conv6
  ret fp128 %add7
}

; Passing a mix of float128 and vector parameters.
; Function Attrs: norecurse nounwind
define void @mixParam_03(fp128 %f1, double* nocapture %d1, <4 x i32> %vec1,
; CHECK-LABEL: mixParam_03:
; CHECK:       # %bb.0: # %entry
; CHECK-DAG:     ld 3, 104(1)
; CHECK-DAG:     mtvsrwa [[REG2:[0-9]+]], 10
; CHECK-DAG:     stxv 2, 0(9)
; CHECK-DAG:     xscvsdqp [[REG1:[0-9]+]], [[REG2]]
; CHECK:         stxvx 3, 0, 3
; CHECK-NEXT:    lxv 2, 0(9)
; CHECK-NEXT:    xsaddqp 2, 2, [[REG1]]
; CHECK-NEXT:    xscvqpdp 2, 2
; CHECK-NEXT:    stxsd 2, 0(5)
; CHECK-NEXT:    blr
                         fp128* nocapture %f2, i32 signext %i1, i8 zeroext %c1,
                         <4 x i32>* nocapture %vec2) {
entry:
  store fp128 %f1, fp128* %f2, align 16
  store <4 x i32> %vec1, <4 x i32>* %vec2, align 16
  %0 = load fp128, fp128* %f2, align 16
  %conv = sitofp i32 %i1 to fp128
  %add = fadd fp128 %0, %conv
  %conv1 = fptrunc fp128 %add to double
  store double %conv1, double* %d1, align 8
  ret void
}

; Function Attrs: norecurse nounwind
define fastcc void @mixParam_03f(fp128 %f1, double* nocapture %d1, <4 x i32> %vec1,
; CHECK-LABEL: mixParam_03f:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    mtvsrwa [[REG0:[0-9]+]], 5
; CHECK-NEXT:    stxv [[REG1:[0-9]+]], 0(4)
; CHECK-NEXT:    stxv [[REG2:[0-9]+]], 0(7)
; CHECK-NEXT:    lxv [[REG1]], 0(4)
; CHECK-NEXT:    xscvsdqp [[REG3:[0-9]+]], [[REG0]]
; CHECK-NEXT:    xsaddqp [[REG4:[0-9]+]], [[REG1]], [[REG3]]
; CHECK-NEXT:    xscvqpdp 2, [[REG4]]
; CHECK-NEXT:    stxsd 2, 0(3)
; CHECK-NEXT:    blr
                                 fp128* nocapture %f2, i32 signext %i1, i8 zeroext %c1,
                                 <4 x i32>* nocapture %vec2) {
entry:
  store fp128 %f1, fp128* %f2, align 16
  store <4 x i32> %vec1, <4 x i32>* %vec2, align 16
  %0 = load fp128, fp128* %f2, align 16
  %conv = sitofp i32 %i1 to fp128
  %add = fadd fp128 %0, %conv
  %conv1 = fptrunc fp128 %add to double
  store double %conv1, double* %d1, align 8
  ret void
}
