; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown -ppc-vsr-nums-as-vr \
; RUN:   -verify-machineinstrs -ppc-asm-full-reg-names < %s | FileCheck %s

; Function Attrs: norecurse nounwind readnone
define fp128 @loadConstant() {
; CHECK-LABEL: loadConstant:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r[[REG0:[0-9]+]], r2, .LCPI0_0@toc@ha
; CHECK-NEXT:    addi r[[REG0]], r[[REG0]], .LCPI0_0@toc@l
; CHECK-NEXT:    lxvx v2, 0, r[[REG0]]
; CHECK-NEXT:    blr
  entry:
    ret fp128 0xL00000000000000004001400000000000
}

; Function Attrs: norecurse nounwind readnone
define fp128 @loadConstant2(fp128 %a, fp128 %b) {
; CHECK-LABEL: loadConstant2:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    addis r[[REG0:[0-9]+]], r2, .LCPI1_0@toc@ha
; CHECK-NEXT:    addi r[[REG0]], r[[REG0]], .LCPI1_0@toc@l
; CHECK-NEXT:    lxvx v[[REG1:[0-9]+]], 0, r[[REG0]]
; CHECK-NEXT:    xsaddqp v2, v2, v[[REG1]]
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
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    extsw r3, r3
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
; CHECK-NEXT:    xsaddqp v2, v2, v3
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
; CHECK-NEXT:    sldi r4, r4, 4
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    add [[REG:r[0-9]+]], r3, r4
; CHECK-NEXT:    lxv v3, -16([[REG]])
; CHECK-NEXT:    xsaddqp v2, v2, v3
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
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    lxv v[[REG0:[0-9]+]], 224(r1)
; CHECK-NEXT:    xsaddqp v2, v2, v4
; CHECK-NEXT:    xsaddqp v2, v2, v5
; CHECK-NEXT:    xsaddqp v2, v2, v6
; CHECK-NEXT:    xsaddqp v2, v2, v7
; CHECK-NEXT:    xsaddqp v2, v2, v8
; CHECK-NEXT:    xsaddqp v2, v2, v9
; CHECK-NEXT:    xsaddqp v2, v2, v10
; CHECK-NEXT:    xsaddqp v2, v2, v11
; CHECK-NEXT:    xsaddqp v2, v2, v12
; CHECK-NEXT:    xsaddqp v2, v2, v13
; CHECK-NEXT:    xssubqp v2, v2, v[[REG0]]
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
; CHECK-DAG:     mtvsrwa [[REG1:v[0-9]+]], r5
; CHECK-DAG:     xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvsdqp v[[REG0:[0-9]+]], [[REG1]]
; CHECK-NEXT:    xsaddqp v2, v2, v[[REG0]]
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
; CHECK-DAG:     mtvsrwa v[[REG0:[0-9]+]], r3
; CHECK-DAG:     xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvsdqp v[[REG1:[0-9]+]], v[[REG0]]
; CHECK-NEXT:    xsaddqp v2, v2, v[[REG1]]
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
; CHECK:         lwz r3, 96(r1)
; CHECK:         add r4, r7, r9
; CHECK:         xscpsgndp v[[REG0:[0-9]+]], f1, f1
; CHECK:         add r4, r4, r10
; CHECK:         xscvdpqp v[[REG0]], v[[REG0]]
; CHECK:         add r3, r4, r3
; CHECK:         clrldi r3, r3, 32
; CHECK:         std r3, 0(r6)
; CHECK:         lxv v[[REG1:[0-9]+]], 0(r8)
; CHECK:         xsaddqp v2, v[[REG1]], v2
; CHECK:         xsaddqp v2, v2, v3
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
; CHECK-NEXT:    add r4, r4, r6
; CHECK-NEXT:    xscpsgndp v[[REG0:[0-9]+]], f1, f1
; CHECK-NEXT:    add r4, r4, r7
; CHECK-NEXT:    xscvdpqp v[[REG0]], v[[REG0]]
; CHECK-NEXT:    add r4, r4, r8
; CHECK-NEXT:    clrldi r4, r4, 32
; CHECK-DAG:     std r4, 0(r3)
; CHECK-DAG:     lxv v[[REG1:[0-9]+]], 0(r5)
; CHECK-NEXT:    xsaddqp v2, v[[REG1]], v2
; CHECK-NEXT:    xsaddqp v2, v2, v[[REG0]] 
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
; CHECK-DAG:     ld r3, 104(r1)
; CHECK-DAG:     stxv v2, 0(r9)
; CHECK:         stxvx v3, 0, r3
; CHECK:         mtvsrwa v[[REG2:[0-9]+]], r10
; CHECK-DAG:     xscvsdqp v[[REG1:[0-9]+]], v[[REG2]]
; CHECK-DAG:     lxv v2, 0(r9)
; CHECK-NEXT:    xsaddqp v2, v2, v[[REG1]]
; CHECK-NEXT:    xscvqpdp v2, v2
; CHECK-NEXT:    stxsd v2, 0(r5)
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
; CHECK-DAG:     mtvsrwa v[[REG0:[0-9]+]], r5
; CHECK-DAG:     stxv v[[REG1:[0-9]+]], 0(r4)
; CHECK-DAG:     stxv v[[REG2:[0-9]+]], 0(r7)
; CHECK-DAG:     lxv v[[REG1]], 0(r4)
; CHECK-NEXT:    xscvsdqp v[[REG3:[0-9]+]], v[[REG0]]
; CHECK-NEXT:    xsaddqp v[[REG4:[0-9]+]], v[[REG1]], v[[REG3]]
; CHECK-NEXT:    xscvqpdp v2, v[[REG4]]
; CHECK-NEXT:    stxsd v2, 0(r3)
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
