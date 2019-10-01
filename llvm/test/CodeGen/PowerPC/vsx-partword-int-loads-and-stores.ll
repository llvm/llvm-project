; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s \
; RUN:   --check-prefix=CHECK-BE
; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecucuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %0, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecucuc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecucuc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <8 x i16> @vecusuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = zext i8 %0 to i16
  %splat.splatinsert = insertelement <8 x i16> undef, i16 %conv, i32 0
  %splat.splat = shufflevector <8 x i16> %splat.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %splat.splat
; CHECK-LABEL: vecusuc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vsplth 2, 2, 3
; CHECK-BE-LABEL: vecusuc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vsplth 2, 2, 3
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecuiuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = zext i8 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecuiuc
; CHECK: lxsibzx 0, 0, 3
; CHECK-NEXT: xxspltw 34, 0, 1
; CHECK-BE-LABEL: vecuiuc
; CHECK-BE: lxsibzx 0, 0, 3
; CHECK-BE-NEXT: xxspltw 34, 0, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @veculuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = zext i8 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: veculuc
; CHECK: lxsibzx 0, 0, 3
; CHECK-NEXT: xxspltd 34, 0, 0
; CHECK-BE-LABEL: veculuc
; CHECK-BE: lxsibzx 0, 0, 3
; CHECK-BE-NEXT: xxspltd 34, 0, 0
}

; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecscuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %0, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecscuc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecscuc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <8 x i16> @vecssuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = zext i8 %0 to i16
  %splat.splatinsert = insertelement <8 x i16> undef, i16 %conv, i32 0
  %splat.splat = shufflevector <8 x i16> %splat.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %splat.splat
; CHECK-LABEL: vecssuc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vsplth 2, 2, 3
; CHECK-BE-LABEL: vecssuc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vsplth 2, 2, 3
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecsiuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = zext i8 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecsiuc
; CHECK: lxsibzx 0, 0, 3
; CHECK-NEXT: xxspltw 34, 0, 1
; CHECK-BE-LABEL: vecsiuc
; CHECK-BE: lxsibzx 0, 0, 3
; CHECK-BE-NEXT: xxspltw 34, 0, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @vecsluc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = zext i8 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: vecsluc
; CHECK: lxsibzx 0, 0, 3
; CHECK-NEXT: xxspltd 34, 0, 0
; CHECK-BE-LABEL: vecsluc
; CHECK-BE: lxsibzx 0, 0, 3
; CHECK-BE-NEXT: xxspltd 34, 0, 0
}

; Function Attrs: norecurse nounwind readonly
define <4 x float> @vecfuc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = uitofp i8 %0 to float
  %splat.splatinsert = insertelement <4 x float> undef, float %conv, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
; CHECK-LABEL: vecfuc
; CHECK: lxsibzx [[LD:[0-9]+]], 0, 3
; CHECK-NEXT: xscvuxdsp [[CONVD:[0-9]+]], [[LD]]
; CHECK-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-NEXT: xxspltw 34, [[CONVS]], 0
; CHECK-BE-LABEL: vecfuc
; CHECK-BE: lxsibzx [[LD:[0-9]+]], 0, 3
; CHECK-BE-NEXT: xscvuxdsp [[CONVD:[0-9]+]], [[LD]]
; CHECK-BE-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-BE-NEXT: xxspltw 34, [[CONVS]], 0
}

; Function Attrs: norecurse nounwind readonly
define <2 x double> @vecduc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = uitofp i8 %0 to double
  %splat.splatinsert = insertelement <2 x double> undef, double %conv, i32 0
  %splat.splat = shufflevector <2 x double> %splat.splatinsert, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %splat.splat
; CHECK-LABEL: vecduc
; CHECK: lxsibzx [[LD:[0-9]+]], 0, 3
; CHECK-NEXT: xscvuxddp [[CONVD:[0-9]+]], [[LD]]
; CHECK-NEXT: xxspltd  34, [[CONVD]], 0
; CHECK-BE-LABEL: vecduc
; CHECK-BE: lxsibzx [[LD:[0-9]+]], 0, 3
; CHECK-BE-NEXT: xscvuxddp [[CONVD:[0-9]+]], [[LD]]
; CHECK-BE-NEXT: xxspltd  34, [[CONVD]], 0
}

; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecucsc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %0, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecucsc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecucsc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecuisc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sext i8 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecuisc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vextsb2w 2, 2
; CHECK-NEXT: xxspltw 34, 34, 1
; CHECK-BE-LABEL: vecuisc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vextsb2w 2, 2
; CHECK-BE-NEXT: xxspltw 34, 34, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @veculsc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sext i8 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: veculsc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vextsb2d 2, 2
; CHECK-NEXT: xxspltd 34, 34, 0
; CHECK-BE-LABEL: veculsc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vextsb2d 2, 2
; CHECK-BE-NEXT: xxspltd 34, 34, 0
}

; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecscsc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %0, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecscsc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecscsc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecsisc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sext i8 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecsisc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vextsb2w 2, 2
; CHECK-NEXT: xxspltw 34, 34, 1
; CHECK-BE-LABEL: vecsisc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vextsb2w 2, 2
; CHECK-BE-NEXT: xxspltw 34, 34, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @vecslsc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sext i8 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: vecslsc
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vextsb2d 2, 2
; CHECK-NEXT: xxspltd 34, 34, 0
; CHECK-BE-LABEL: vecslsc
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vextsb2d 2, 2
; CHECK-BE-NEXT: xxspltd 34, 34, 0
}

; Function Attrs: norecurse nounwind readonly
define <4 x float> @vecfsc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sitofp i8 %0 to float
  %splat.splatinsert = insertelement <4 x float> undef, float %conv, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
; CHECK-LABEL: vecfsc
; CHECK: lxsibzx
; CHECK-NEXT: vextsb2d
; CHECK-NEXT: xscvsxdsp [[CONVD:[0-9]+]],
; CHECK-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-NEXT: xxspltw 34, [[CONVS]], 0
; CHECK-BE-LABEL: vecfsc
; CHECK-BE: lxsibzx [[LD:[0-9]+]], 0, 3
; CHECK-BE-NEXT: vextsb2d
; CHECK-BE-NEXT: xscvsxdsp [[CONVD:[0-9]+]],
; CHECK-BE-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-BE-NEXT: xxspltw 34, [[CONVS]], 0
}

; Function Attrs: norecurse nounwind readonly
define <2 x double> @vecdsc(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sitofp i8 %0 to double
  %splat.splatinsert = insertelement <2 x double> undef, double %conv, i32 0
  %splat.splat = shufflevector <2 x double> %splat.splatinsert, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %splat.splat
; CHECK-LABEL: vecdsc
; CHECK: lxsibzx
; CHECK-NEXT: vextsb2d
; CHECK-NEXT: xscvsxddp [[CONVD:[0-9]+]],
; CHECK-NEXT: xxspltd  34, [[CONVD]], 0
; CHECK-BE-LABEL: vecdsc
; CHECK-BE: lxsibzx
; CHECK-BE-NEXT: vextsb2d
; CHECK-BE-NEXT: xscvsxddp [[CONVD:[0-9]+]],
; CHECK-BE-NEXT: xxspltd  34, [[CONVD]], 0
}

; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecucus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = trunc i16 %0 to i8
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %conv, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecucus
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecucus
; CHECK-BE: addi [[OFFSET:[0-9]+]], [[OFFSET]], 1
; CHECK-BE-NEXT: lxsibzx 34, 0, [[OFFSET]]
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <8 x i16> @vecusus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %splat.splatinsert = insertelement <8 x i16> undef, i16 %0, i32 0
  %splat.splat = shufflevector <8 x i16> %splat.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %splat.splat
; CHECK-LABEL: vecusus
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vsplth 2, 2, 3
; CHECK-BE-LABEL: vecusus
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vsplth 2, 2, 3
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecuius(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = zext i16 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecuius
; CHECK: lxsihzx 0, 0, 3
; CHECK-NEXT: xxspltw 34, 0, 1
; CHECK-BE-LABEL: vecuius
; CHECK-BE: lxsihzx 0, 0, 3
; CHECK-BE-NEXT: xxspltw 34, 0, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @veculus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = zext i16 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: veculus
; CHECK: lxsihzx 0, 0, 3
; CHECK-NEXT: xxspltd 34, 0, 0
; CHECK-BE-LABEL: veculus
; CHECK-BE: lxsihzx 0, 0, 3
; CHECK-BE-NEXT: xxspltd 34, 0, 0
}

; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecscus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = trunc i16 %0 to i8
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %conv, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecscus
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecscus
; CHECK-BE: addi [[OFFSET:[0-9]+]], [[OFFSET]], 1
; CHECK-BE-NEXT: lxsibzx 34, 0, [[OFFSET]]
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <8 x i16> @vecssus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %splat.splatinsert = insertelement <8 x i16> undef, i16 %0, i32 0
  %splat.splat = shufflevector <8 x i16> %splat.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %splat.splat
; CHECK-LABEL: vecssus
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vsplth 2, 2, 3
; CHECK-BE-LABEL: vecssus
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vsplth 2, 2, 3
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecsius(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = zext i16 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecsius
; CHECK: lxsihzx 0, 0, 3
; CHECK-NEXT: xxspltw 34, 0, 1
; CHECK-BE-LABEL: vecsius
; CHECK-BE: lxsihzx 0, 0, 3
; CHECK-BE-NEXT: xxspltw 34, 0, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @vecslus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = zext i16 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: vecslus
; CHECK: lxsihzx 0, 0, 3
; CHECK-NEXT: xxspltd 34, 0, 0
; CHECK-BE-LABEL: vecslus
; CHECK-BE: lxsihzx 0, 0, 3
; CHECK-BE-NEXT: xxspltd 34, 0, 0
}

; Function Attrs: norecurse nounwind readonly
define <4 x float> @vecfus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = uitofp i16 %0 to float
  %splat.splatinsert = insertelement <4 x float> undef, float %conv, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
; CHECK-LABEL: vecfus
; CHECK: lxsihzx [[LD:[0-9]+]], 0, 3
; CHECK-NEXT: xscvuxdsp [[CONVD:[0-9]+]], [[LD]]
; CHECK-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-NEXT: xxspltw 34, [[CONVS]], 0
; CHECK-BE-LABEL: vecfus
; CHECK-BE: lxsihzx [[LD:[0-9]+]], 0, 3
; CHECK-BE-NEXT: xscvuxdsp [[CONVD:[0-9]+]], [[LD]]
; CHECK-BE-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-BE-NEXT: xxspltw 34, [[CONVS]], 0
}

; Function Attrs: norecurse nounwind readonly
define <2 x double> @vecdus(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = uitofp i16 %0 to double
  %splat.splatinsert = insertelement <2 x double> undef, double %conv, i32 0
  %splat.splat = shufflevector <2 x double> %splat.splatinsert, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %splat.splat
; CHECK-LABEL: vecdus
; CHECK: lxsihzx [[LD:[0-9]+]], 0, 3
; CHECK-NEXT: xscvuxddp [[CONVD:[0-9]+]], [[LD]]
; CHECK-NEXT: xxspltd  34, [[CONVD]], 0
; CHECK-BE-LABEL: vecdus
; CHECK-BE: lxsihzx [[LD:[0-9]+]], 0, 3
; CHECK-BE-NEXT: xscvuxddp [[CONVD:[0-9]+]], [[LD]]
; CHECK-BE-NEXT: xxspltd  34, [[CONVD]], 0
}

; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecucss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = trunc i16 %0 to i8
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %conv, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecucss
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecucss
; CHECK-BE: addi [[OFFSET:[0-9]+]], [[OFFSET]], 1
; CHECK-BE-NEXT: lxsibzx 34, 0, [[OFFSET]]
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecuiss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sext i16 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecuiss
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vextsh2w 2, 2
; CHECK-NEXT: xxspltw 34, 34, 1
; CHECK-BE-LABEL: vecuiss
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vextsh2w 2, 2
; CHECK-BE-NEXT: xxspltw 34, 34, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @veculss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sext i16 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: veculss
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vextsh2d 2, 2
; CHECK-NEXT: xxspltd 34, 34, 0
; CHECK-BE-LABEL: veculss
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vextsh2d 2, 2
; CHECK-BE-NEXT: xxspltd 34, 34, 0
}

; Function Attrs: norecurse nounwind readonly
define <16 x i8> @vecscss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = trunc i16 %0 to i8
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %conv, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK-LABEL: vecscss
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vspltb 2, 2, 7
; CHECK-BE-LABEL: vecscss
; CHECK-BE: addi [[OFFSET:[0-9]+]], [[OFFSET]], 1
; CHECK-BE-NEXT: lxsibzx 34, 0, [[OFFSET]]
; CHECK-BE-NEXT: vspltb 2, 2, 7
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @vecsiss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sext i16 %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK-LABEL: vecsiss
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vextsh2w 2, 2
; CHECK-NEXT: xxspltw 34, 34, 1
; CHECK-BE-LABEL: vecsiss
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vextsh2w 2, 2
; CHECK-BE-NEXT: xxspltw 34, 34, 1
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @vecslss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sext i16 %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK-LABEL: vecslss
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vextsh2d 2, 2
; CHECK-NEXT: xxspltd 34, 34, 0
; CHECK-BE-LABEL: vecslss
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vextsh2d 2, 2
; CHECK-BE-NEXT: xxspltd 34, 34, 0
}

; Function Attrs: norecurse nounwind readonly
define <4 x float> @vecfss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sitofp i16 %0 to float
  %splat.splatinsert = insertelement <4 x float> undef, float %conv, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
; CHECK-LABEL: vecfss
; CHECK: lxsihzx
; CHECK-NEXT: vextsh2d
; CHECK-NEXT: xscvsxdsp [[CONVD:[0-9]+]],
; CHECK-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-NEXT: xxspltw 34, [[CONVS]], 0
; CHECK-BE-LABEL: vecfss
; CHECK-BE: lxsihzx [[LD:[0-9]+]], 0, 3
; CHECK-BE-NEXT: vextsh2d
; CHECK-BE-NEXT: xscvsxdsp [[CONVD:[0-9]+]],
; CHECK-BE-NEXT: xscvdpspn [[CONVS:[0-9]+]], [[CONVD]]
; CHECK-BE-NEXT: xxspltw 34, [[CONVS]], 0
}

; Function Attrs: norecurse nounwind readonly
define <2 x double> @vecdss(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sitofp i16 %0 to double
  %splat.splatinsert = insertelement <2 x double> undef, double %conv, i32 0
  %splat.splat = shufflevector <2 x double> %splat.splatinsert, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %splat.splat
; CHECK-LABEL: vecdss
; CHECK: lxsihzx
; CHECK-NEXT: vextsh2d
; CHECK-NEXT: xscvsxddp [[CONVD:[0-9]+]],
; CHECK-NEXT: xxspltd  34, [[CONVD]], 0
; CHECK-BE-LABEL: vecdss
; CHECK-BE: lxsihzx
; CHECK-BE-NEXT: vextsh2d
; CHECK-BE-NEXT: xscvsxddp [[CONVD:[0-9]+]],
; CHECK-BE-NEXT: xxspltd  34, [[CONVD]], 0
}

; Function Attrs: norecurse nounwind
define void @storefsc(float %f, i8* nocapture %ptr) {
entry:
  %conv = fptosi float %f to i8
  store i8 %conv, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storefsc
; CHECK: xscvdpsxws 0, 1
; CHECK: stxsibx 0, 0, 4
; CHECK-BE-LABEL: storefsc
; CHECK-BE: xscvdpsxws 0, 1
; CHECK-BE: stxsibx 0, 0, 4
}

; Function Attrs: norecurse nounwind
define void @storedsc(double %d, i8* nocapture %ptr) {
entry:
  %conv = fptosi double %d to i8
  store i8 %conv, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storedsc
; CHECK: xscvdpsxws 0, 1
; CHECK: stxsibx 0, 0, 4
; CHECK-BE-LABEL: storedsc
; CHECK-BE: xscvdpsxws 0, 1
; CHECK-BE: stxsibx 0, 0, 4
}

; Function Attrs: norecurse nounwind
define void @storevcsc0(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 0
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc0
; CHECK: vsldoi 2, 2, 2, 8
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc0
; CHECK-BE: vsldoi 2, 2, 2, 9
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc1(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 1
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc1
; CHECK: vsldoi 2, 2, 2, 7
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc1
; CHECK-BE: vsldoi 2, 2, 2, 10
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc2(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 2
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc2
; CHECK: vsldoi 2, 2, 2, 6
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc2
; CHECK-BE: vsldoi 2, 2, 2, 11
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc3(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 3
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc3
; CHECK: vsldoi 2, 2, 2, 5
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc3
; CHECK-BE: vsldoi 2, 2, 2, 12
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc4(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 4
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc4
; CHECK: vsldoi 2, 2, 2, 4
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc4
; CHECK-BE: vsldoi 2, 2, 2, 13
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc5(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 5
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc5
; CHECK: vsldoi 2, 2, 2, 3
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc5
; CHECK-BE: vsldoi 2, 2, 2, 14
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc6(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 6
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc6
; CHECK: vsldoi 2, 2, 2, 2
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc6
; CHECK-BE: vsldoi 2, 2, 2, 15
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc7(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 7
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc7
; CHECK: vsldoi 2, 2, 2, 1
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc7
; CHECK-BE: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc8(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 8
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc8
; CHECK: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc8
; CHECK-BE: vsldoi 2, 2, 2, 1
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc9(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 9
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc9
; CHECK: vsldoi 2, 2, 2, 15
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc9
; CHECK-BE: vsldoi 2, 2, 2, 2
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc10(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 10
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc10
; CHECK: vsldoi 2, 2, 2, 14
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc10
; CHECK-BE: vsldoi 2, 2, 2, 3
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc11(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 11
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc11
; CHECK: vsldoi 2, 2, 2, 13
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc11
; CHECK-BE: vsldoi 2, 2, 2, 4
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc12(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 12
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc12
; CHECK: vsldoi 2, 2, 2, 12
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc12
; CHECK-BE: vsldoi 2, 2, 2, 5
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc13(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 13
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc13
; CHECK: vsldoi 2, 2, 2, 11
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc13
; CHECK-BE: vsldoi 2, 2, 2, 6
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc14(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 14
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc14
; CHECK: vsldoi 2, 2, 2, 10
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc14
; CHECK-BE: vsldoi 2, 2, 2, 7
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevcsc15(<16 x i8> %v, i8* nocapture %ptr) {
entry:
  %vecext = extractelement <16 x i8> %v, i32 15
  store i8 %vecext, i8* %ptr, align 1
  ret void
; CHECK-LABEL: storevcsc15
; CHECK: vsldoi 2, 2, 2, 9
; CHECK-NEXT: stxsibx 34, 0, 5
; CHECK-BE-LABEL: storevcsc15
; CHECK-BE: vsldoi 2, 2, 2, 8
; CHECK-BE-NEXT: stxsibx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storefss(float %f, i16* nocapture %ptr) {
entry:
  %conv = fptosi float %f to i16
  store i16 %conv, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storefss
; CHECK: xscvdpsxws 0, 1
; CHECK: stxsihx 0, 0, 4
; CHECK-BE-LABEL: storefss
; CHECK-BE: xscvdpsxws 0, 1
; CHECK-BE: stxsihx 0, 0, 4
}

; Function Attrs: norecurse nounwind
define void @storedss(double %d, i16* nocapture %ptr) {
entry:
  %conv = fptosi double %d to i16
  store i16 %conv, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storedss
; CHECK: xscvdpsxws 0, 1
; CHECK: stxsihx 0, 0, 4
; CHECK-BE-LABEL: storedss
; CHECK-BE: xscvdpsxws 0, 1
; CHECK-BE: stxsihx 0, 0, 4
}

; Function Attrs: norecurse nounwind
define void @storevsss0(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 0
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss0
; CHECK: vsldoi 2, 2, 2, 8
; CHECK-NEXT: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss0
; CHECK-BE: vsldoi 2, 2, 2, 10
; CHECK-BE-NEXT: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevsss1(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 1
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss1
; CHECK: vsldoi 2, 2, 2, 6
; CHECK-NEXT: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss1
; CHECK-BE: vsldoi 2, 2, 2, 12
; CHECK-BE-NEXT: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevsss2(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 2
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss2
; CHECK: vsldoi 2, 2, 2, 4
; CHECK-NEXT: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss2
; CHECK-BE: vsldoi 2, 2, 2, 14
; CHECK-BE-NEXT: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevsss3(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 3
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss3
; CHECK: vsldoi 2, 2, 2, 2
; CHECK-NEXT: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss3
; CHECK-BE: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevsss4(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 4
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss4
; CHECK: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss4
; CHECK-BE: vsldoi 2, 2, 2, 2
; CHECK-BE-NEXT: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevsss5(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 5
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss5
; CHECK: vsldoi 2, 2, 2, 14
; CHECK-NEXT: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss5
; CHECK-BE: vsldoi 2, 2, 2, 4
; CHECK-BE-NEXT: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevsss6(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 6
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss6
; CHECK: vsldoi 2, 2, 2, 12
; CHECK-NEXT: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss6
; CHECK-BE: vsldoi 2, 2, 2, 6
; CHECK-BE-NEXT: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind
define void @storevsss7(<8 x i16> %v, i16* nocapture %ptr) {
entry:
  %vecext = extractelement <8 x i16> %v, i32 7
  store i16 %vecext, i16* %ptr, align 2
  ret void
; CHECK-LABEL: storevsss7
; CHECK: vsldoi 2, 2, 2, 10
; CHECK-NEXT: stxsihx 34, 0, 5
; CHECK-BE-LABEL: storevsss7
; CHECK-BE: vsldoi 2, 2, 2, 8
; CHECK-BE-NEXT: stxsihx 34, 0, 5
}

; Function Attrs: norecurse nounwind readonly
define float @convscf(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sitofp i8 %0 to float
  ret float %conv
; CHECK-LABEL: convscf
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vextsb2d 2, 2
; CHECK-NEXT: xscvsxdsp 1, 34
; CHECK-BE-LABEL: convscf
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vextsb2d 2, 2
; CHECK-BE-NEXT: xscvsxdsp 1, 34
}

; Function Attrs: norecurse nounwind readonly
define float @convucf(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = uitofp i8 %0 to float
  ret float %conv
; CHECK-LABEL: convucf
; CHECK: lxsibzx 0, 0, 3
; CHECK-NEXT: xscvuxdsp 1, 0
; CHECK-BE-LABEL: convucf
; CHECK-BE: lxsibzx 0, 0, 3
; CHECK-BE-NEXT: xscvuxdsp 1, 0
}

; Function Attrs: norecurse nounwind readonly
define double @convscd(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = sitofp i8 %0 to double
; CHECK-LABEL: convscd
; CHECK: lxsibzx 34, 0, 3
; CHECK-NEXT: vextsb2d 2, 2
; CHECK-NEXT: xscvsxddp 1, 34
; CHECK-BE-LABEL: convscd
; CHECK-BE: lxsibzx 34, 0, 3
; CHECK-BE-NEXT: vextsb2d 2, 2
; CHECK-BE-NEXT: xscvsxddp 1, 34
  ret double %conv
}

; Function Attrs: norecurse nounwind readonly
define double @convucd(i8* nocapture readonly %ptr) {
entry:
  %0 = load i8, i8* %ptr, align 1
  %conv = uitofp i8 %0 to double
  ret double %conv
; CHECK-LABEL: convucd
; CHECK: lxsibzx 0, 0, 3
; CHECK-NEXT: xscvuxddp 1, 0
; CHECK-BE-LABEL: convucd
; CHECK-BE: lxsibzx 0, 0, 3
; CHECK-BE-NEXT: xscvuxddp 1, 0
}

; Function Attrs: norecurse nounwind readonly
define float @convssf(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sitofp i16 %0 to float
  ret float %conv
; CHECK-LABEL: convssf
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vextsh2d 2, 2
; CHECK-NEXT: xscvsxdsp 1, 34
; CHECK-BE-LABEL: convssf
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vextsh2d 2, 2
; CHECK-BE-NEXT: xscvsxdsp 1, 34
}

; Function Attrs: norecurse nounwind readonly
define float @convusf(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = uitofp i16 %0 to float
  ret float %conv
; CHECK-LABEL: convusf
; CHECK: lxsihzx 0, 0, 3
; CHECK-NEXT: xscvuxdsp 1, 0
; CHECK-BE-LABEL: convusf
; CHECK-BE: lxsihzx 0, 0, 3
; CHECK-BE-NEXT: xscvuxdsp 1, 0
}

; Function Attrs: norecurse nounwind readonly
define double @convssd(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = sitofp i16 %0 to double
  ret double %conv
; CHECK-LABEL: convssd
; CHECK: lxsihzx 34, 0, 3
; CHECK-NEXT: vextsh2d 2, 2
; CHECK-NEXT: xscvsxddp 1, 34
; CHECK-BE-LABEL: convssd
; CHECK-BE: lxsihzx 34, 0, 3
; CHECK-BE-NEXT: vextsh2d 2, 2
; CHECK-BE-NEXT: xscvsxddp 1, 34
}

; Function Attrs: norecurse nounwind readonly
define double @convusd(i16* nocapture readonly %ptr) {
entry:
  %0 = load i16, i16* %ptr, align 2
  %conv = uitofp i16 %0 to double
  ret double %conv
; CHECK-LABEL: convusd
; CHECK: lxsihzx 0, 0, 3
; CHECK-NEXT: xscvuxddp 1, 0
; CHECK-BE-LABEL: convusd
; CHECK-BE: lxsihzx 0, 0, 3
; CHECK-BE-NEXT: xscvuxddp 1, 0
}
