; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -ppc-vsr-nums-as-vr < %s | FileCheck %s

@mem = global [5 x i64] [i64 56, i64 63, i64 3, i64 5, i64 6], align 8
@umem = global [5 x i64] [i64 560, i64 100, i64 34, i64 2, i64 5], align 8
@swMem = global [5 x i32] [i32 5, i32 2, i32 3, i32 4, i32 0], align 4

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
define void @sdwConv2qp_testXForm(fp128* nocapture %sink,
                                  i8* nocapture readonly %a) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %a, i64 3
  %0 = bitcast i8* %add.ptr to i64*
  %1 = load i64, i64* %0, align 8
  %conv = sitofp i64 %1 to fp128
  store fp128 %conv, fp128* %sink, align 16
  ret void

; CHECK-LABEL: sdwConv2qp_testXForm
; CHECK: addi [[REG:[0-9]+]], 4, 3
; CHECK-NEXT: lxsd [[REG1:[0-9]+]], 0([[REG]])
; CHECK-NEXT: xscvsdqp [[CONV:[0-9]+]], [[REG1]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @udwConv2qp_testXForm(fp128* nocapture %sink,
                                  i8* nocapture readonly %a) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %a, i64 3
  %0 = bitcast i8* %add.ptr to i64*
  %1 = load i64, i64* %0, align 8
  %conv = uitofp i64 %1 to fp128
  store fp128 %conv, fp128* %sink, align 16
  ret void

; CHECK-LABEL: udwConv2qp_testXForm
; CHECK: addi [[REG:[0-9]+]], 4, 3
; CHECK-NEXT: lxsd [[REG1:[0-9]+]], 0([[REG]])
; CHECK-NEXT: xscvudqp [[CONV:[0-9]+]], [[REG1]]
; CHECK-NEXT: stxv [[CONV]], 0(3)
; CHECK-NEXT: blr
}
