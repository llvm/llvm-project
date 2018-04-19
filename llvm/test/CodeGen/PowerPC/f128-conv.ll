; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -ppc-vsr-nums-as-vr \
; RUN:   -verify-machineinstrs < %s | FileCheck %s

@mem = global [5 x i64] [i64 56, i64 63, i64 3, i64 5, i64 6], align 8
@umem = global [5 x i64] [i64 560, i64 100, i64 34, i64 2, i64 5], align 8
@swMem = global [5 x i32] [i32 5, i32 2, i32 3, i32 4, i32 0], align 4
@uwMem = global [5 x i32] [i32 5, i32 2, i32 3, i32 4, i32 0], align 4
@uhwMem = local_unnamed_addr global [5 x i16] [i16 5, i16 2, i16 3, i16 4, i16 0], align 2
@ubMem = local_unnamed_addr global [5 x i8] c"\05\02\03\04\00", align 1

; Function Attrs: norecurse nounwind
define void @sdwConv2qp(fp128* nocapture %a, i64 %b) {
entry:
  %conv = sitofp i64 %b to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: sdwConv2qp
; CHECK: mtvsrd [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @sdwConv2qp_02(fp128* nocapture %a) {
entry:
  %0 = load i64, i64* getelementptr inbounds 
                        ([5 x i64], [5 x i64]* @mem, i64 0, i64 2), align 8
  %conv = sitofp i64 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: sdwConv2qp_02
; CHECK: addis [[REG:[0-9]+]], 2, .LC0@toc@ha
; CHECK: ld [[REG]], .LC0@toc@l([[REG]])
; CHECK: lxsd [[REG0:[0-9]+]], 16([[REG]])
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG0]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @sdwConv2qp_03(fp128* nocapture %a, i64* nocapture readonly %b) {
entry:
  %0 = load i64, i64* %b, align 8
  %conv = sitofp i64 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: sdwConv2qp_03
; CHECK-NOT: ld
; CHECK: lxsd [[REG0:[0-9]+]], 0(4)
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG0]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @udwConv2qp(fp128* nocapture %a, i64 %b) {
entry:
  %conv = uitofp i64 %b to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: udwConv2qp
; CHECK: mtvsrd [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @udwConv2qp_02(fp128* nocapture %a) {
entry:
  %0 = load i64, i64* getelementptr inbounds
                        ([5 x i64], [5 x i64]* @umem, i64 0, i64 4), align 8
  %conv = uitofp i64 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: udwConv2qp_02
; CHECK: addis [[REG:[0-9]+]], 2, .LC1@toc@ha
; CHECK: ld [[REG]], .LC1@toc@l([[REG]])
; CHECK: lxsd [[REG0:[0-9]+]], 32([[REG]])
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG0]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @udwConv2qp_03(fp128* nocapture %a, i64* nocapture readonly %b) {
entry:
  %0 = load i64, i64* %b, align 8
  %conv = uitofp i64 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: udwConv2qp_03
; CHECK-NOT: ld
; CHECK: lxsd [[REG:[0-9]+]], 0(4)
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define fp128* @sdwConv2qp_testXForm(fp128* returned %sink,
                                    i8* nocapture readonly %a) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %a, i64 73333
  %0 = bitcast i8* %add.ptr to i64*
  %1 = load i64, i64* %0, align 8
  %conv = sitofp i64 %1 to fp128
  store fp128 %conv, fp128* %sink, align 16
  ret fp128* %sink

; CHECK-LABEL: sdwConv2qp_testXForm
; CHECK: lxsdx [[REG:[0-9]+]],
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define fp128* @udwConv2qp_testXForm(fp128* returned %sink,
                                    i8* nocapture readonly %a) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %a, i64 73333
  %0 = bitcast i8* %add.ptr to i64*
  %1 = load i64, i64* %0, align 8
  %conv = uitofp i64 %1 to fp128
  store fp128 %conv, fp128* %sink, align 16
  ret fp128* %sink

; CHECK-LABEL: udwConv2qp_testXForm
; CHECK: lxsdx [[REG:[0-9]+]],
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @swConv2qp(fp128* nocapture %a, i32 signext %b) {
entry:
  %conv = sitofp i32 %b to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: swConv2qp
; CHECK-NOT: lwz
; CHECK: mtvsrwa [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @swConv2qp_02(fp128* nocapture %a, i32* nocapture readonly %b) {
entry:
  %0 = load i32, i32* %b, align 4
  %conv = sitofp i32 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: swConv2qp_02
; CHECK-NOT: lwz
; CHECK: lxsiwax [[REG:[0-9]+]], 0, 4
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @swConv2qp_03(fp128* nocapture %a) {
entry:
  %0 = load i32, i32* getelementptr inbounds
                        ([5 x i32], [5 x i32]* @swMem, i64 0, i64 3), align 4
  %conv = sitofp i32 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: swConv2qp_03
; CHECK: addis [[REG:[0-9]+]], 2, .LC2@toc@ha
; CHECK: ld [[REG]], .LC2@toc@l([[REG]])
; CHECK: addi [[REG2:[0-9]+]], [[REG]], 12
; CHECK: lxsiwax [[REG0:[0-9]+]], 0, [[REG2]]
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG0]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uwConv2qp(fp128* nocapture %a, i32 zeroext %b) {
entry:
  %conv = uitofp i32 %b to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: uwConv2qp
; CHECK-NOT: lwz
; CHECK: mtvsrwz [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uwConv2qp_02(fp128* nocapture %a, i32* nocapture readonly %b) {
entry:
  %0 = load i32, i32* %b, align 4
  %conv = uitofp i32 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: uwConv2qp_02
; CHECK-NOT: lwz
; CHECK: lxsiwzx [[REG:[0-9]+]], 0, 4
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uwConv2qp_03(fp128* nocapture %a) {
entry:
  %0 = load i32, i32* getelementptr inbounds
                        ([5 x i32], [5 x i32]* @uwMem, i64 0, i64 3), align 4
  %conv = uitofp i32 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: uwConv2qp_03
; CHECK: addis [[REG:[0-9]+]], 2, .LC3@toc@ha
; CHECK-NEXT: ld [[REG]], .LC3@toc@l([[REG]])
; CHECK-NEXT: addi [[REG2:[0-9]+]], [[REG]], 12
; CHECK-NEXT: lxsiwzx [[REG1:[0-9]+]], 0, [[REG2]]
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG1]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uwConv2qp_04(fp128* nocapture %a,
                          i32 zeroext %b, i32* nocapture readonly %c) {
entry:
  %0 = load i32, i32* %c, align 4
  %add = add i32 %0, %b
  %conv = uitofp i32 %add to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: uwConv2qp_04
; CHECK: lwz [[REG:[0-9]+]], 0(5)
; CHECK-NEXT: add [[REG1:[0-9]+]], [[REG]], [[REG1]]
; CHECK-NEXT: mtvsrwz [[REG0:[0-9]+]], [[REG1]]
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG0]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uhwConv2qp(fp128* nocapture %a, i16 zeroext %b) {
entry:
  %conv = uitofp i16 %b to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void


; CHECK-LABEL: uhwConv2qp
; CHECK: mtvsrwz [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uhwConv2qp_02(fp128* nocapture %a, i16* nocapture readonly %b) {
entry:
  %0 = load i16, i16* %b, align 2
  %conv = uitofp i16 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: uhwConv2qp_02
; CHECK: lxsihzx [[REG:[0-9]+]], 0, 4
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uhwConv2qp_03(fp128* nocapture %a) {
entry:
  %0 = load i16, i16* getelementptr inbounds
                        ([5 x i16], [5 x i16]* @uhwMem, i64 0, i64 3), align 2
  %conv = uitofp i16 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: uhwConv2qp_03
; CHECK: addis [[REG0:[0-9]+]], 2, .LC4@toc@ha
; CHECK: ld [[REG0]], .LC4@toc@l([[REG0]])
; CHECK: addi [[REG0]], [[REG0]], 6
; CHECK: lxsihzx [[REG:[0-9]+]], 0, [[REG0]]
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @uhwConv2qp_04(fp128* nocapture %a, i16 zeroext %b,
                           i16* nocapture readonly %c) {
entry:
  %conv = zext i16 %b to i32
  %0 = load i16, i16* %c, align 2
  %conv1 = zext i16 %0 to i32
  %add = add nuw nsw i32 %conv1, %conv
  %conv2 = sitofp i32 %add to fp128
  store fp128 %conv2, fp128* %a, align 16
  ret void

; CHECK-LABEL: uhwConv2qp_04
; CHECK: lhz [[REG0:[0-9]+]], 0(5)
; CHECK: add 4, [[REG0]], 4
; CHECK: mtvsrwa [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @ubConv2qp(fp128* nocapture %a, i8 zeroext %b) {
entry:
  %conv = uitofp i8 %b to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: ubConv2qp
; CHECK: mtvsrwz [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @ubConv2qp_02(fp128* nocapture %a, i8* nocapture readonly %b) {
entry:
  %0 = load i8, i8* %b, align 1
  %conv = uitofp i8 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: ubConv2qp_02
; CHECK: lxsibzx [[REG:[0-9]+]], 0, 4
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @ubConv2qp_03(fp128* nocapture %a) {
entry:
  %0 = load i8, i8* getelementptr inbounds 
                      ([5 x i8], [5 x i8]* @ubMem, i64 0, i64 2), align 1
  %conv = uitofp i8 %0 to fp128
  store fp128 %conv, fp128* %a, align 16
  ret void

; CHECK-LABEL: ubConv2qp_03
; CHECK: addis [[REG0:[0-9]+]], 2, .LC5@toc@ha
; CHECK: ld [[REG0]], .LC5@toc@l([[REG0]])
; CHECK: addi [[REG0]], [[REG0]], 2
; CHECK: lxsibzx [[REG:[0-9]+]], 0, [[REG0]]
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @ubConv2qp_04(fp128* nocapture %a, i8 zeroext %b,
                          i8* nocapture readonly %c) {
entry:
  %conv = zext i8 %b to i32
  %0 = load i8, i8* %c, align 1
  %conv1 = zext i8 %0 to i32
  %add = add nuw nsw i32 %conv1, %conv
  %conv2 = sitofp i32 %add to fp128
  store fp128 %conv2, fp128* %a, align 16
  ret void

; CHECK-LABEL: ubConv2qp_04
; CHECK: lbz [[REG0:[0-9]+]], 0(5)
; CHECK: add 4, [[REG0]], 4
; CHECK: mtvsrwa [[REG:[0-9]+]], 4
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}
