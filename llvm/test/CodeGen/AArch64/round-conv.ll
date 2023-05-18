; RUN: llc < %s -mtriple=arm64 | FileCheck %s

; CHECK-LABEL: testmsws:
; CHECK: fcvtms w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testmsws(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxs:
; CHECK: fcvtms x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testmsxs(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testmswd:
; CHECK: fcvtms w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testmswd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testmsxd:
; CHECK: fcvtms x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testmsxd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testmuws:
; CHECK: fcvtmu w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testmuws(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testmuxs:
; CHECK: fcvtmu x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testmuxs(float %a) {
entry:
  %call = call float @floorf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testmuwd:
; CHECK: fcvtmu w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testmuwd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testmuxd:
; CHECK: fcvtmu x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testmuxd(double %a) {
entry:
  %call = call double @floor(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testpsws:
; CHECK: fcvtps w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testpsws(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testpsxs:
; CHECK: fcvtps x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testpsxs(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testpswd:
; CHECK: fcvtps w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testpswd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testpsxd:
; CHECK: fcvtps x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testpsxd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testpuws:
; CHECK: fcvtpu w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testpuws(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testpuxs:
; CHECK: fcvtpu x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testpuxs(float %a) {
entry:
  %call = call float @ceilf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testpuwd:
; CHECK: fcvtpu w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testpuwd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testpuxd:
; CHECK: fcvtpu x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testpuxd(double %a) {
entry:
  %call = call double @ceil(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testzsws:
; CHECK: fcvtzs w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testzsws(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testzsxs:
; CHECK: fcvtzs x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testzsxs(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testzswd:
; CHECK: fcvtzs w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testzswd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testzsxd:
; CHECK: fcvtzs x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testzsxd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testzuws:
; CHECK: fcvtzu w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testzuws(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testzuxs:
; CHECK: fcvtzu x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testzuxs(float %a) {
entry:
  %call = call float @truncf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testzuwd:
; CHECK: fcvtzu w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testzuwd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testzuxd:
; CHECK: fcvtzu x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testzuxd(double %a) {
entry:
  %call = call double @trunc(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testasws:
; CHECK: fcvtas w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testasws(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptosi float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testasxs:
; CHECK: fcvtas x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testasxs(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptosi float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testaswd:
; CHECK: fcvtas w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testaswd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptosi double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testasxd:
; CHECK: fcvtas x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testasxd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptosi double %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testauws:
; CHECK: fcvtau w0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i32 @testauws(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptoui float %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testauxs:
; CHECK: fcvtau x0, s0
; CHECK-NOT: frintx {{s[0-9]+}}, s0
define i64 @testauxs(float %a) {
entry:
  %call = call float @roundf(float %a) nounwind readnone
  %conv = fptoui float %call to i64
  ret i64 %conv
}

; CHECK-LABEL: testauwd:
; CHECK: fcvtau w0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i32 @testauwd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptoui double %call to i32
  ret i32 %conv
}

; CHECK-LABEL: testauxd:
; CHECK: fcvtau x0, d0
; CHECK-NOT: frintx {{d[0-9]+}}, d0
define i64 @testauxd(double %a) {
entry:
  %call = call double @round(double %a) nounwind readnone
  %conv = fptoui double %call to i64
  ret i64 %conv
}

declare float @floorf(float) nounwind readnone
declare double @floor(double) nounwind readnone
declare float @ceilf(float) nounwind readnone
declare double @ceil(double) nounwind readnone
declare float @truncf(float) nounwind readnone
declare double @trunc(double) nounwind readnone
declare float @roundf(float) nounwind readnone
declare double @round(double) nounwind readnone
