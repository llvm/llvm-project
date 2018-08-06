; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown -ppc-vsr-nums-as-vr \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown -ppc-vsr-nums-as-vr \
; RUN:   -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK-PWR8 %s

; ==========================================
; Tests for store of fp_to_sint converstions
; ==========================================

; Function Attrs: norecurse nounwind
define void @dpConv2sdw(double* nocapture readonly %a, i64* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i64
  store i64 %conv, i64* %b, align 8
  ret void

; CHECK-LABEL: dpConv2sdw
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsd [[CONV]], 0(4)
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2sdw
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2sw(double* nocapture readonly %a, i32* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i32
  store i32 %conv, i32* %b, align 4
  ret void

; CHECK-LABEL: dpConv2sw
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2sw
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2shw(double* nocapture readonly %a, i16* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i16
  store i16 %conv, i16* %b, align 2
  ret void

; CHECK-LABEL: dpConv2shw
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2shw
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sth [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2sb(double* nocapture readonly %a, i8* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i8
  store i8 %conv, i8* %b, align 1
  ret void

; CHECK-LABEL: dpConv2sb
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2sb
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stb [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2sdw(float* nocapture readonly %a, i64* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i64
  store i64 %conv, i64* %b, align 8
  ret void

; CHECK-LABEL: spConv2sdw
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsd [[CONV]], 0(4)
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2sdw
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2sw(float* nocapture readonly %a, i32* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i32
  store i32 %conv, i32* %b, align 4
  ret void

; CHECK-LABEL: spConv2sw
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2sw
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2shw(float* nocapture readonly %a, i16* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i16
  store i16 %conv, i16* %b, align 2
  ret void

; CHECK-LABEL: spConv2shw
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2shw
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sth [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2sb(float* nocapture readonly %a, i8* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i8
  store i8 %conv, i8* %b, align 1
  ret void

; CHECK-LABEL: spConv2sb
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2sb
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stb [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2sdw_x(double* nocapture readonly %a, i64* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i64
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: dpConv2sdw_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 3
; CHECK-NEXT: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2sdw_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8: sldi [[REG:[0-9]+]], 5, 3
; CHECK-PWR8-NEXT: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2sw_x(double* nocapture readonly %a, i32* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i32
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 %conv, i32* %arrayidx, align 4
  ret void

; CHECK-LABEL: dpConv2sw_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2sw_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2shw_x(double* nocapture readonly %a, i16* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i16
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i16, i16* %b, i64 %idxprom
  store i16 %conv, i16* %arrayidx, align 2
  ret void

; CHECK-LABEL: dpConv2shw_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 1
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2shw_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 1
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sthx [[REG]], 4, 5
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2sb_x(double* nocapture readonly %a, i8* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptosi double %0 to i8
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i8, i8* %b, i64 %idxprom
  store i8 %conv, i8* %arrayidx, align 1
  ret void

; CHECK-LABEL: dpConv2sb_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 4, 5
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2sb_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stbx [[REG]], 4, 5
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2sdw_x(float* nocapture readonly %a, i64* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i64
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: spConv2sdw_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 3
; CHECK-NEXT: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2sdw_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 3
; CHECK-PWR8-NEXT: xscvdpsxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2sw_x(float* nocapture readonly %a, i32* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i32
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 %conv, i32* %arrayidx, align 4
  ret void

; CHECK-LABEL: spConv2sw_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2sw_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2shw_x(float* nocapture readonly %a, i16* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i16
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i16, i16* %b, i64 %idxprom
  store i16 %conv, i16* %arrayidx, align 2
  ret void

; CHECK-LABEL: spConv2shw_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK: sldi [[REG:[0-9]+]], 5, 1
; CHECK: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2shw_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 1
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG2:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sthx [[REG2]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2sb_x(float* nocapture readonly %a, i8* nocapture %b,
                          i32 signext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptosi float %0 to i8
  %idxprom = sext i32 %idx to i64
  %arrayidx = getelementptr inbounds i8, i8* %b, i64 %idxprom
  store i8 %conv, i8* %arrayidx, align 1
  ret void

; CHECK-LABEL: spConv2sb_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 4, 5
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2sb_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stbx [[REG]], 4, 5
; CHECK-PWR8-NEXT: blr
}

; ==========================================
; Tests for store of fp_to_uint converstions
; ==========================================

; Function Attrs: norecurse nounwind
define void @dpConv2udw(double* nocapture readonly %a, i64* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i64
  store i64 %conv, i64* %b, align 8
  ret void

; CHECK-LABEL: dpConv2udw
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsd [[CONV]], 0(4)
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2udw
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2uw(double* nocapture readonly %a, i32* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i32
  store i32 %conv, i32* %b, align 4
  ret void

; CHECK-LABEL: dpConv2uw
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2uw
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2uhw(double* nocapture readonly %a, i16* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i16
  store i16 %conv, i16* %b, align 2
  ret void

; CHECK-LABEL: dpConv2uhw
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2uhw
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sth [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2ub(double* nocapture readonly %a, i8* nocapture %b) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i8
  store i8 %conv, i8* %b, align 1
  ret void

; CHECK-LABEL: dpConv2ub
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2ub
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stb [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2udw(float* nocapture readonly %a, i64* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i64
  store i64 %conv, i64* %b, align 8
  ret void

; CHECK-LABEL: spConv2udw
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsd [[CONV]], 0(4)
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2udw
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2uw(float* nocapture readonly %a, i32* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i32
  store i32 %conv, i32* %b, align 4
  ret void

; CHECK-LABEL: spConv2uw
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2uw
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 0, 4
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2uhw(float* nocapture readonly %a, i16* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i16
  store i16 %conv, i16* %b, align 2
  ret void

; CHECK-LABEL: spConv2uhw
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2uhw
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sth [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2ub(float* nocapture readonly %a, i8* nocapture %b) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i8
  store i8 %conv, i8* %b, align 1
  ret void

; CHECK-LABEL: spConv2ub
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 0, 4
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2ub
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stb [[REG]], 0(4)
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2udw_x(double* nocapture readonly %a, i64* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i64
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: dpConv2udw_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 3
; CHECK-NEXT: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2udw_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8: sldi [[REG:[0-9]+]], 5, 3
; CHECK-PWR8-NEXT: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2uw_x(double* nocapture readonly %a, i32* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i32
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 %conv, i32* %arrayidx, align 4
  ret void

; CHECK-LABEL: dpConv2uw_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2uw_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-PWR8-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2uhw_x(double* nocapture readonly %a, i16* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i16
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i16, i16* %b, i64 %idxprom
  store i16 %conv, i16* %arrayidx, align 2
  ret void

; CHECK-LABEL: dpConv2uhw_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 1
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2uhw_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 1
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sthx [[REG]], 4, 5
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @dpConv2ub_x(double* nocapture readonly %a, i8* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load double, double* %a, align 8
  %conv = fptoui double %0 to i8
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i8, i8* %b, i64 %idxprom
  store i8 %conv, i8* %arrayidx, align 1
  ret void

; CHECK-LABEL: dpConv2ub_x
; CHECK: lfd [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 4, 5
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: dpConv2ub_x
; CHECK-PWR8: lfdx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stbx [[REG]], 4, 5
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2udw_x(float* nocapture readonly %a, i64* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i64
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i64, i64* %b, i64 %idxprom
  store i64 %conv, i64* %arrayidx, align 8
  ret void

; CHECK-LABEL: spConv2udw_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 3
; CHECK-NEXT: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2udw_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 3
; CHECK-PWR8-NEXT: xscvdpuxds [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stxsdx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2uw_x(float* nocapture readonly %a, i32* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i32
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i32, i32* %b, i64 %idxprom
  store i32 %conv, i32* %arrayidx, align 4
  ret void

; CHECK-LABEL: spConv2uw_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2uw_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 2
; CHECK-PWR8-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: stfiwx [[CONV]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2uhw_x(float* nocapture readonly %a, i16* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i16
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i16, i16* %b, i64 %idxprom
  store i16 %conv, i16* %arrayidx, align 2
  ret void

; CHECK-LABEL: spConv2uhw_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK: sldi [[REG:[0-9]+]], 5, 1
; CHECK: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsihx [[CONV]], 4, [[REG]]
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2uhw_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: sldi [[REG:[0-9]+]], 5, 1
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG2:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: sthx [[REG2]], 4, [[REG]]
; CHECK-PWR8-NEXT: blr
}

; Function Attrs: norecurse nounwind
define void @spConv2ub_x(float* nocapture readonly %a, i8* nocapture %b,
                          i32 zeroext %idx) {
entry:
  %0 = load float, float* %a, align 4
  %conv = fptoui float %0 to i8
  %idxprom = zext i32 %idx to i64
  %arrayidx = getelementptr inbounds i8, i8* %b, i64 %idxprom
  store i8 %conv, i8* %arrayidx, align 1
  ret void

; CHECK-LABEL: spConv2ub_x
; CHECK: lfs [[LD:[0-9]+]], 0(3)
; CHECK-NEXT: xscvdpuxws [[CONV:[0-9]+]], [[LD]]
; CHECK-NEXT: stxsibx [[CONV]], 4, 5
; CHECK-NEXT: blr

; CHECK-PWR8-LABEL: spConv2ub_x
; CHECK-PWR8: lfsx [[LD:[0-9]+]], 0, 3
; CHECK-PWR8-NEXT: xscvdpsxws [[CONV:[0-9]+]], [[LD]]
; CHECK-PWR8-NEXT: mfvsrwz [[REG:[0-9]+]], [[CONV]]
; CHECK-PWR8-NEXT: stbx [[REG]], 4, 5
; CHECK-PWR8-NEXT: blr
}
