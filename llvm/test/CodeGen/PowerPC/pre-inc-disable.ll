; RUN: llc -mcpu=pwr9 -O3 -verify-machineinstrs -ppc-vsr-nums-as-vr \
; RUN:     -ppc-asm-full-reg-names -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:     < %s | FileCheck %s

; RUN: llc -mcpu=pwr9 -O3 -verify-machineinstrs -ppc-vsr-nums-as-vr \
; RUN:     -ppc-asm-full-reg-names -mtriple=powerpc64-unknown-linux-gnu \
; RUN:     < %s | FileCheck %s --check-prefix=P9BE

; Function Attrs: norecurse nounwind readonly
define signext i32 @test_pre_inc_disable_1(i8* nocapture readonly %pix1, i32 signext %i_stride_pix1, i8* nocapture readonly %pix2) {
; CHECK-LABEL: test_pre_inc_disable_1:
; CHECK:   # %bb.0: # %entry
; CHECK:    lfd f0, 0(r5)
; CHECK:    addis r5, r2
; CHECK:    addi r5, r5,
; CHECK:    lxvx v2, 0, r5
; CHECK:    addis r5, r2,
; CHECK:    addi r5, r5,
; CHECK:    lxvx v4, 0, r5
; CHECK:    xxpermdi v5, f0, f0, 2
; CHECK:    xxlxor v3, v3, v3
; CHECK-DAG: vperm v[[VR1:[0-9]+]], v5, v3, v4
; CHECK-DAG: vperm v[[VR2:[0-9]+]], v3, v5, v2
; CHECK-DAG: xvnegsp v[[VR3:[0-9]+]], v[[VR1]]
; CHECK-DAG: xvnegsp v[[VR4:[0-9]+]], v[[VR2]]

; CHECK:  .LBB0_1: # %for.cond1.preheader
; CHECK:    lfd f0, 0(r3)
; CHECK:    xxpermdi v1, f0, f0, 2
; CHECK:    vperm v6, v3, v1, v2
; CHECK:    vperm v1, v1, v3, v4
; CHECK-DAG:    xvnegsp v6, v6
; CHECK-DAG:    xvnegsp v1, v1
; CHECK-DAG: vabsduw v1, v1, v[[VR3]]
; CHECK-DAG: vabsduw v6, v6, v[[VR4]]
; CHECK:    vadduwm v1, v1, v6
; CHECK:    xxswapd v6, v1
; CHECK:    vadduwm v1, v1, v6
; CHECK:    xxspltw v6, v1, 2
; CHECK:    vadduwm v1, v1, v6
; CHECK:    vextuwrx r7, r5, v1
; CHECK:    ldux r8, r3, r4
; CHECK:    add r3, r3, r4
; CHECK:    add r6, r7, r6
; CHECK:    mtvsrd f0, r8
; CHECK:    xxswapd v1, vs0
; CHECK:    vperm v6, v3, v1, v2
; CHECK:    vperm v1, v1, v3, v4
; CHECK-DAG: xvnegsp v6, v6
; CHECK-DAG: xvnegsp v1, v1
; CHECK-DAG: vabsduw v1, v1, v[[VR3]]
; CHECK-DAG: vabsduw v6, v6, v[[VR4]]
; CHECK:    vadduwm v1, v1, v6
; CHECK:    xxswapd v6, v1
; CHECK:    vadduwm v1, v1, v6
; CHECK:    xxspltw v6, v1, 2
; CHECK:    vadduwm v1, v1, v6
; CHECK:    vextuwrx r7, r5, v1
; CHECK:    add r6, r7, r6
; CHECK:    bdnz .LBB0_1
; CHECK:    extsw r3, r6
; CHECK:    blr

; P9BE-LABEL: test_pre_inc_disable_1:
; P9BE:    lfd f0, 0(r5)
; P9BE:    addis r5, r2,
; P9BE:    addi r5, r5,
; P9BE:    lxvx v2, 0, r5
; P9BE:    addis r5, r2,
; P9BE:    addi r5, r5,
; P9BE:    lxvx v4, 0, r5
; P9BE:    xxlor v5, vs0, vs0
; P9BE:    xxlxor v3, v3, v3
; P9BE-DAG: li r5, 0
; P9BE-DAG: vperm v[[VR1:[0-9]+]], v3, v5, v2
; P9BE-DAG: vperm v[[VR2:[0-9]+]], v3, v5, v4
; P9BE-DAG: xvnegsp v[[VR3:[0-9]+]], v[[VR1]]
; P9BE-DAG: xvnegsp v[[VR4:[0-9]+]], v[[VR2]]

; P9BE:  .LBB0_1: # %for.cond1.preheader
; P9BE:    lfd f0, 0(r3)
; P9BE:    xxlor v1, vs0, vs0
; P9BE:    vperm v6, v3, v1, v4
; P9BE:    vperm v1, v3, v1, v2
; P9BE-DAG: xvnegsp v6, v6
; P9BE-DAG: xvnegsp v1, v1
; P9BE-DAG: vabsduw v1, v1, v[[VR3]]
; P9BE-DAG: vabsduw v6, v6, v[[VR4]]
; P9BE:    vadduwm v1, v6, v1
; P9BE:    xxswapd v6, v1
; P9BE:    vadduwm v1, v1, v6
; P9BE:    xxspltw v6, v1, 1
; P9BE:    vadduwm v1, v1, v6
; P9BE:    vextuwlx r[[GR1:[0-9]+]], r5, v1
; P9BE:    add r6, r[[GR1]], r6
; P9BE:    ldux r[[GR2:[0-9]+]], r3, r4
; P9BE:    add r3, r3, r4
; P9BE:    mtvsrd v1, r[[GR2]]
; P9BE:    vperm v6, v3, v1, v2
; P9BE:    vperm v1, v3, v1, v4
; P9BE-DAG: xvnegsp v6, v6
; P9BE-DAG: xvnegsp v1, v1
; P9BE-DAG: vabsduw v1, v1, v[[VR4]]
; P9BE-DAG: vabsduw v6, v6, v[[VR3]]
; P9BE:    vadduwm v1, v1, v6
; P9BE:    xxswapd v6, v1
; P9BE:    vadduwm v1, v1, v6
; P9BE:    xxspltw v6, v1, 1
; P9BE:    vadduwm v1, v1, v6
; P9BE:    vextuwlx r7, r5, v1
; P9BE:    add r6, r7, r6
; P9BE:    bdnz .LBB0_1
; P9BE:    extsw r3, r6
; P9BE:    blr
entry:
  %idx.ext = sext i32 %i_stride_pix1 to i64
  %0 = bitcast i8* %pix2 to <8 x i8>*
  %1 = load <8 x i8>, <8 x i8>* %0, align 1
  %2 = zext <8 x i8> %1 to <8 x i32>
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader, %entry
  %y.024 = phi i32 [ 0, %entry ], [ %inc9.1, %for.cond1.preheader ]
  %i_sum.023 = phi i32 [ 0, %entry ], [ %op.extra.1, %for.cond1.preheader ]
  %pix1.addr.022 = phi i8* [ %pix1, %entry ], [ %add.ptr.1, %for.cond1.preheader ]
  %3 = bitcast i8* %pix1.addr.022 to <8 x i8>*
  %4 = load <8 x i8>, <8 x i8>* %3, align 1
  %5 = zext <8 x i8> %4 to <8 x i32>
  %6 = sub nsw <8 x i32> %5, %2
  %7 = icmp slt <8 x i32> %6, zeroinitializer
  %8 = sub nsw <8 x i32> zeroinitializer, %6
  %9 = select <8 x i1> %7, <8 x i32> %8, <8 x i32> %6
  %rdx.shuf = shufflevector <8 x i32> %9, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add nsw <8 x i32> %9, %rdx.shuf
  %rdx.shuf32 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx33 = add nsw <8 x i32> %bin.rdx, %rdx.shuf32
  %rdx.shuf34 = shufflevector <8 x i32> %bin.rdx33, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx35 = add nsw <8 x i32> %bin.rdx33, %rdx.shuf34
  %10 = extractelement <8 x i32> %bin.rdx35, i32 0
  %op.extra = add nsw i32 %10, %i_sum.023
  %add.ptr = getelementptr inbounds i8, i8* %pix1.addr.022, i64 %idx.ext
  %11 = bitcast i8* %add.ptr to <8 x i8>*
  %12 = load <8 x i8>, <8 x i8>* %11, align 1
  %13 = zext <8 x i8> %12 to <8 x i32>
  %14 = sub nsw <8 x i32> %13, %2
  %15 = icmp slt <8 x i32> %14, zeroinitializer
  %16 = sub nsw <8 x i32> zeroinitializer, %14
  %17 = select <8 x i1> %15, <8 x i32> %16, <8 x i32> %14
  %rdx.shuf.1 = shufflevector <8 x i32> %17, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx.1 = add nsw <8 x i32> %17, %rdx.shuf.1
  %rdx.shuf32.1 = shufflevector <8 x i32> %bin.rdx.1, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx33.1 = add nsw <8 x i32> %bin.rdx.1, %rdx.shuf32.1
  %rdx.shuf34.1 = shufflevector <8 x i32> %bin.rdx33.1, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx35.1 = add nsw <8 x i32> %bin.rdx33.1, %rdx.shuf34.1
  %18 = extractelement <8 x i32> %bin.rdx35.1, i32 0
  %op.extra.1 = add nsw i32 %18, %op.extra
  %add.ptr.1 = getelementptr inbounds i8, i8* %add.ptr, i64 %idx.ext
  %inc9.1 = add nuw nsw i32 %y.024, 2
  %exitcond.1 = icmp eq i32 %inc9.1, 8
  br i1 %exitcond.1, label %for.cond.cleanup, label %for.cond1.preheader

for.cond.cleanup:                                 ; preds = %for.cond1.preheader
  ret i32 %op.extra.1
}

; Function Attrs: norecurse nounwind readonly
define signext i32 @test_pre_inc_disable_2(i8* nocapture readonly %pix1, i8* nocapture readonly %pix2) {
; CHECK-LABEL: test_pre_inc_disable_2:
; CHECK:    lfd f0, 0(r3)
; CHECK:    addis r3, r2,
; CHECK:    addi r3, r3, .LCPI1_0@toc@l
; CHECK:    lxvx v4, 0, r3
; CHECK:    addis r3, r2,
; CHECK:    xxpermdi v2, f0, f0, 2
; CHECK:    lfd f0, 0(r4)
; CHECK:    addi r3, r3, .LCPI1_1@toc@l
; CHECK:    xxlxor v3, v3, v3
; CHECK:    lxvx v0, 0, r3
; CHECK:    xxpermdi v1, f0, f0, 2
; CHECK:    vperm v5, v2, v3, v4
; CHECK:    vperm v2, v3, v2, v0
; CHECK:    vperm v0, v3, v1, v0
; CHECK:    vperm v3, v1, v3, v4
; CHECK:    vabsduw v2, v2, v0
; CHECK:    vabsduw v3, v5, v3
; CHECK:    vadduwm v2, v3, v2
; CHECK:    xxswapd v3, v2
; CHECK:    vadduwm v2, v2, v3
; CHECK:    xxspltw v3, v2, 2
; CHECK:    vadduwm v2, v2, v3
; CHECK:    vextuwrx r3, r3, v2
; CHECK:    extsw r3, r3
; CHECK:    blr

; P9BE-LABEL: test_pre_inc_disable_2:
; P9BE:    lfd f0, 0(r3)
; P9BE:    addis r3, r2,
; P9BE:    addi r3, r3,
; P9BE:    lxvx v4, 0, r3
; P9BE:    addis r3, r2,
; P9BE:    addi r3, r3,
; P9BE:    xxlor v2, vs0, vs0
; P9BE:    lfd f0, 0(r4)
; P9BE:    lxvx v0, 0, r3
; P9BE:    xxlxor v3, v3, v3
; P9BE:    xxlor v1, vs0, vs0
; P9BE:    vperm v5, v3, v2, v4
; P9BE:    vperm v2, v3, v2, v0
; P9BE:    vperm v0, v3, v1, v0
; P9BE:    vperm v3, v3, v1, v4
; P9BE:    vabsduw v2, v2, v0
; P9BE:    vabsduw v3, v5, v3
; P9BE:    vadduwm v2, v3, v2
; P9BE:    xxswapd v3, v2
; P9BE:    vadduwm v2, v2, v3
; P9BE:    xxspltw v3, v2, 1
; P9BE:    vadduwm v2, v2, v3
; P9BE:    vextuwlx r3, r3, v2
; P9BE:    extsw r3, r3
; P9BE:    blr
entry:
  %0 = bitcast i8* %pix1 to <8 x i8>*
  %1 = load <8 x i8>, <8 x i8>* %0, align 1
  %2 = zext <8 x i8> %1 to <8 x i32>
  %3 = bitcast i8* %pix2 to <8 x i8>*
  %4 = load <8 x i8>, <8 x i8>* %3, align 1
  %5 = zext <8 x i8> %4 to <8 x i32>
  %6 = sub nsw <8 x i32> %2, %5
  %7 = icmp slt <8 x i32> %6, zeroinitializer
  %8 = sub nsw <8 x i32> zeroinitializer, %6
  %9 = select <8 x i1> %7, <8 x i32> %8, <8 x i32> %6
  %rdx.shuf = shufflevector <8 x i32> %9, <8 x i32> undef, <8 x i32> <i32 4, i32 5, i32 6, i32 7, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx = add nsw <8 x i32> %9, %rdx.shuf
  %rdx.shuf12 = shufflevector <8 x i32> %bin.rdx, <8 x i32> undef, <8 x i32> <i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx13 = add nsw <8 x i32> %bin.rdx, %rdx.shuf12
  %rdx.shuf14 = shufflevector <8 x i32> %bin.rdx13, <8 x i32> undef, <8 x i32> <i32 1, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef, i32 undef>
  %bin.rdx15 = add nsw <8 x i32> %bin.rdx13, %rdx.shuf14
  %10 = extractelement <8 x i32> %bin.rdx15, i32 0
  ret i32 %10
}


; Generated from C source:
;
;#include <stdint.h>
;#include <stdlib.h>
;int test_pre_inc_disable_1( uint8_t *pix1, int i_stride_pix1, uint8_t *pix2 ) {
;    int i_sum = 0;
;    for( int y = 0; y < 8; y++ ) {
;        for( int x = 0; x < 8; x++) {
;            i_sum += abs( pix1[x] - pix2[x] )
;        }
;        pix1 += i_stride_pix1;
;    }
;    return i_sum;
;}

;int test_pre_inc_disable_2( uint8_t *pix1, uint8_t *pix2 ) {
;  int i_sum = 0;
;  for( int x = 0; x < 8; x++ ) {
;    i_sum += abs( pix1[x] - pix2[x] );
;  }
;
;  return i_sum;
;}

