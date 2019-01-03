; RUN: llc -relocation-model=pic -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -verify-machineinstrs -enable-ppc-quad-precision \
; RUN:   -ppc-vsr-nums-as-vr -ppc-asm-full-reg-names < %s | FileCheck %s

@f128Array = global [4 x fp128] [fp128 0xL00000000000000004004C00000000000,
                                 fp128 0xLF000000000000000400808AB851EB851,
                                 fp128 0xL5000000000000000400E0C26324C8366,
                                 fp128 0xL8000000000000000400A24E2E147AE14],
                                align 16

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2sdw(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptosi fp128 %0 to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2sdw
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK-NEXT: xscvqpsdz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: mfvsrd r3, v[[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sdw_02(i64* nocapture %res) local_unnamed_addr #1 {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptosi fp128 %0 to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2sdw_02
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK: lxv v[[REG:[0-9]+]], 32(r[[REG0]])
; CHECK-NEXT: xscvqpsdz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: stxsd v[[CONV]], 0(r3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2sdw_03(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2sdw_03
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK: lxv v[[REG1:[0-9]+]], 16(r[[REG0]])
; CHECK: xsaddqp v[[REG]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpsdz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: mfvsrd r3, v[[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sdw_04(fp128* nocapture readonly %a,
                           fp128* nocapture readonly %b, i64* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2sdw_04
; CHECK-DAG: lxv v[[REG1:[0-9]+]], 0(r4)
; CHECK-DAG: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK: xsaddqp v[[REG]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpsdz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: stxsd v[[CONV]], 0(r5)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sdw_testXForm(i64* nocapture %res, i32 signext %idx) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 2), align 16
  %conv = fptosi fp128 %0 to i64
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %res, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: qpConv2sdw_testXForm
; CHECK: xscvqpsdz v[[CONV:[0-9]+]],
; CHECK: stxsdx v[[CONV]], r3, r4
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2udw(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptoui fp128 %0 to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2udw
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK-NEXT: xscvqpudz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: mfvsrd r3, v[[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2udw_02(i64* nocapture %res) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptoui fp128 %0 to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2udw_02
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK: lxv v[[REG:[0-9]+]], 32(r[[REG0]])
; CHECK-NEXT: xscvqpudz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: stxsd v[[CONV]], 0(r3)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define i64 @qpConv2udw_03(fp128* nocapture readonly %a) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i64
  ret i64 %conv

; CHECK-LABEL: qpConv2udw_03
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK-DAG: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK-DAG: lxv v[[REG1:[0-9]+]], 16(r[[REG0]])
; CHECK: xsaddqp v[[REG]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpudz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: mfvsrd r3, v[[CONV]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2udw_04(fp128* nocapture readonly %a,
                           fp128* nocapture readonly %b, i64* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i64
  store i64 %conv, i64* %res, align 8
  ret void

; CHECK-LABEL: qpConv2udw_04
; CHECK-DAG: lxv v[[REG1:[0-9]+]], 0(r4)
; CHECK-DAG: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK: xsaddqp v[[REG]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpudz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: stxsd v[[CONV]], 0(r5)
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2udw_testXForm(i64* nocapture %res, i32 signext %idx) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 0), align 16
  %conv = fptoui fp128 %0 to i64
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %res, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: qpConv2udw_testXForm
; CHECK: xscvqpudz v[[CONV:[0-9]+]],
; CHECK: stxsdx v[[CONV]], r3, r4
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define signext i32 @qpConv2sw(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptosi fp128 %0 to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2sw
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK-NEXT: xscvqpswz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: mfvsrwz r[[REG2:[0-9]+]], v[[CONV]]
; CHECK-NEXT: extsw r3, r[[REG2]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sw_02(i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptosi fp128 %0 to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2sw_02
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK: lxv v[[REG:[0-9]+]], 32(r[[REG0]])
; CHECK-NEXT: xscvqpswz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: stxsiwx v[[CONV]], 0, r3
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define signext i32 @qpConv2sw_03(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2sw_03
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK-DAG: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK-DAG: lxv v[[REG1:[0-9]+]], 16(r[[REG0]])
; CHECK-NEXT: xsaddqp v[[ADD:[0-9]+]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpswz v[[CONV:[0-9]+]], v[[ADD]]
; CHECK-NEXT: mfvsrwz r[[REG2:[0-9]+]], v[[CONV]]
; CHECK-NEXT: extsw r3, r[[REG2]]
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2sw_04(fp128* nocapture readonly %a,
                          fp128* nocapture readonly %b, i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2sw_04
; CHECK-DAG: lxv v[[REG1:[0-9]+]], 0(r4)
; CHECK-DAG: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK-NEXT: xsaddqp v[[ADD:[0-9]+]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpswz v[[CONV:[0-9]+]], v[[ADD]]
; CHECK-NEXT: stxsiwx v[[CONV]], 0, r5
; CHECK-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define zeroext i32 @qpConv2uw(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptoui fp128 %0 to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2uw
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK-NEXT: xscvqpuwz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: mfvsrwz r3, v[[CONV]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2uw_02(i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 2), align 16
  %conv = fptoui fp128 %0 to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2uw_02
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK: lxv v[[REG:[0-9]+]], 32(r[[REG0]])
; CHECK-NEXT: xscvqpuwz v[[CONV:[0-9]+]], v[[REG]]
; CHECK-NEXT: stxsiwx v[[CONV]], 0, r3
; CHECK: blr
}

; Function Attrs: norecurse nounwind readonly
define zeroext i32 @qpConv2uw_03(fp128* nocapture readonly %a)  {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array, i64 0,
                             i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i32
  ret i32 %conv

; CHECK-LABEL: qpConv2uw_03
; CHECK: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK: addis r[[REG0:[0-9]+]], r2, .LC0@toc@ha
; CHECK-DAG: ld r[[REG0]], .LC0@toc@l(r[[REG0]])
; CHECK-DAG: lxv v[[REG1:[0-9]+]], 16(r[[REG0]])
; CHECK-NEXT: xsaddqp v[[ADD:[0-9]+]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpuwz v[[CONV:[0-9]+]], v[[ADD]]
; CHECK-NEXT: mfvsrwz r3, v[[CONV]]
; CHECK: blr
}

; Function Attrs: norecurse nounwind
define void @qpConv2uw_04(fp128* nocapture readonly %a,
                          fp128* nocapture readonly %b, i32* nocapture %res) {
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i32
  store i32 %conv, i32* %res, align 4
  ret void

; CHECK-LABEL: qpConv2uw_04
; CHECK-DAG: lxv v[[REG1:[0-9]+]], 0(r4)
; CHECK-DAG: lxv v[[REG:[0-9]+]], 0(r3)
; CHECK-NEXT: xsaddqp v[[ADD:[0-9]+]], v[[REG]], v[[REG1]]
; CHECK-NEXT: xscvqpuwz v[[CONV:[0-9]+]], v[[ADD]]
; CHECK-NEXT: stxsiwx v[[CONV]], 0, r5
; CHECK: blr
}

; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py

; Function Attrs: norecurse nounwind readonly
define signext i16 @qpConv2shw(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2shw:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    extsw r3, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptosi fp128 %0 to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2shw_02(i16* nocapture %res) {
; CHECK-LABEL: qpConv2shw_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-NEXT:    lxv v2, 32(r4)
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    stxsihx v2, 0, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 2), align 16
  %conv = fptosi fp128 %0 to i16
  store i16 %conv, i16* %res, align 2
  ret void
}

; Function Attrs: norecurse nounwind readonly
define signext i16 @qpConv2shw_03(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2shw_03:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    addis [[REG:r[0-9]+]], r2, .LC0@toc@ha
; CHECK-NEXT:    ld [[REG1:r[0-9]+]], .LC0@toc@l([[REG]])
; CHECK-NEXT:    lxv v3, 16([[REG1]])
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    extsw r3, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2shw_04(fp128* nocapture readonly %a,
                           fp128* nocapture readonly %b, i16* nocapture %res) {
; CHECK-LABEL: qpConv2shw_04:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    lxv v3, 0(r4)
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    stxsihx v2, 0, r5
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i16
  store i16 %conv, i16* %res, align 2
  ret void
}

; Function Attrs: norecurse nounwind readonly
define zeroext i16 @qpConv2uhw(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2uhw:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    clrldi r3, r3, 32
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptoui fp128 %0 to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2uhw_02(i16* nocapture %res) {
; CHECK-LABEL: qpConv2uhw_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-NEXT:    lxv v2, 32(r4)
; CHECK-NEXT:    xscvqpuwz v2, v2
; CHECK-NEXT:    stxsihx v2, 0, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 2), align 16
  %conv = fptoui fp128 %0 to i16
  store i16 %conv, i16* %res, align 2
  ret void
}

; Function Attrs: norecurse nounwind readonly
define zeroext i16 @qpConv2uhw_03(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2uhw_03:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    addis [[REG:r[0-9]+]], r2, .LC0@toc@ha
; CHECK-NEXT:    ld [[REG1:r[0-9]+]], .LC0@toc@l([[REG]])
; CHECK-NEXT:    lxv v3, 16([[REG1]])
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    clrldi r3, r3, 32
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i16
  ret i16 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2uhw_04(fp128* nocapture readonly %a,
                           fp128* nocapture readonly %b, i16* nocapture %res) {
; CHECK-LABEL: qpConv2uhw_04:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    lxv v3, 0(r4)
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpuwz v2, v2
; CHECK-NEXT:    stxsihx v2, 0, r5
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i16
  store i16 %conv, i16* %res, align 2
  ret void
}

; Function Attrs: norecurse nounwind readonly
define signext i8 @qpConv2sb(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2sb:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    extsw r3, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptosi fp128 %0 to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2sb_02(i8* nocapture %res) {
; CHECK-LABEL: qpConv2sb_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-NEXT:    lxv v2, 32(r4)
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    stxsibx v2, 0, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 2), align 16
  %conv = fptosi fp128 %0 to i8
  store i8 %conv, i8* %res, align 1
  ret void
}

; Function Attrs: norecurse nounwind readonly
define signext i8 @qpConv2sb_03(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2sb_03:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    addis [[REG:r[0-9]+]], r2, .LC0@toc@ha
; CHECK-NEXT:    ld [[REG1:r[0-9]+]], .LC0@toc@l([[REG]])
; CHECK-NEXT:    lxv v3, 16([[REG1]])
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    extsw r3, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2sb_04(fp128* nocapture readonly %a,
                          fp128* nocapture readonly %b, i8* nocapture %res) {
; CHECK-LABEL: qpConv2sb_04:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    lxv v3, 0(r4)
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    stxsibx v2, 0, r5
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptosi fp128 %add to i8
  store i8 %conv, i8* %res, align 1
  ret void
}

; Function Attrs: norecurse nounwind readonly
define zeroext i8 @qpConv2ub(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2ub:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    clrldi r3, r3, 32
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %conv = fptoui fp128 %0 to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2ub_02(i8* nocapture %res) {
; CHECK-LABEL: qpConv2ub_02:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    addis r4, r2, .LC0@toc@ha
; CHECK-NEXT:    ld r4, .LC0@toc@l(r4)
; CHECK-NEXT:    lxv v2, 32(r4)
; CHECK-NEXT:    xscvqpuwz v2, v2
; CHECK-NEXT:    stxsibx v2, 0, r3
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 2), align 16
  %conv = fptoui fp128 %0 to i8
  store i8 %conv, i8* %res, align 1
  ret void
}

; Function Attrs: norecurse nounwind readonly
define zeroext i8 @qpConv2ub_03(fp128* nocapture readonly %a) {
; CHECK-LABEL: qpConv2ub_03:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    addis [[REG:r[0-9]+]], r2, .LC0@toc@ha
; CHECK-NEXT:    ld [[REG1:r[0-9]+]], .LC0@toc@l([[REG]])
; CHECK-NEXT:    lxv v3, 16([[REG1]])
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpswz v2, v2
; CHECK-NEXT:    mfvsrwz r3, v2
; CHECK-NEXT:    clrldi r3, r3, 32
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* getelementptr inbounds
                            ([4 x fp128], [4 x fp128]* @f128Array,
                             i64 0, i64 1), align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i8
  ret i8 %conv
}

; Function Attrs: norecurse nounwind
define void @qpConv2ub_04(fp128* nocapture readonly %a,
                          fp128* nocapture readonly %b, i8* nocapture %res) {
; CHECK-LABEL: qpConv2ub_04:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lxv v2, 0(r3)
; CHECK-NEXT:    lxv v3, 0(r4)
; CHECK-NEXT:    xsaddqp v2, v2, v3
; CHECK-NEXT:    xscvqpuwz v2, v2
; CHECK-NEXT:    stxsibx v2, 0, r5
; CHECK-NEXT:    blr
entry:
  %0 = load fp128, fp128* %a, align 16
  %1 = load fp128, fp128* %b, align 16
  %add = fadd fp128 %0, %1
  %conv = fptoui fp128 %add to i8
  store i8 %conv, i8* %res, align 1
  ret void
}
