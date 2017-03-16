; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s

; Function Attrs: nounwind
define zeroext i8 @_Z6testcff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptoui float %0 to i8
  ret i8 %conv
; CHECK-LABEL: @_Z6testcff
; CHECK: xscvdpsxws [[CONVREG01:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG01]]
}

; Function Attrs: nounwind
define float @_Z6testfcc(i8 zeroext %arg) {
entry:
  %arg.addr = alloca i8, align 1
  store i8 %arg, i8* %arg.addr, align 1
  %0 = load i8, i8* %arg.addr, align 1
  %conv = uitofp i8 %0 to float
  ret float %conv
; CHECK-LABEL: @_Z6testfcc
; CHECK: mtvsrwz [[MOVEREG01:[0-9]+]], 3
; CHECK: xscvuxdsp 1, [[MOVEREG01]]
}

; Function Attrs: nounwind
define zeroext i8 @_Z6testcdd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptoui double %0 to i8
  ret i8 %conv
; CHECK-LABEL: @_Z6testcdd
; CHECK: xscvdpsxws [[CONVREG02:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG02]]
}

; Function Attrs: nounwind
define double @_Z6testdcc(i8 zeroext %arg) {
entry:
  %arg.addr = alloca i8, align 1
  store i8 %arg, i8* %arg.addr, align 1
  %0 = load i8, i8* %arg.addr, align 1
  %conv = uitofp i8 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z6testdcc
; CHECK: mtvsrwz [[MOVEREG02:[0-9]+]], 3
; CHECK: xscvuxddp 1, [[MOVEREG02]]
}

; Function Attrs: nounwind
define zeroext i8 @_Z7testucff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptoui float %0 to i8
  ret i8 %conv
; CHECK-LABEL: @_Z7testucff
; CHECK: xscvdpsxws [[CONVREG03:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG03]]
}

; Function Attrs: nounwind
define float @_Z7testfuch(i8 zeroext %arg) {
entry:
  %arg.addr = alloca i8, align 1
  store i8 %arg, i8* %arg.addr, align 1
  %0 = load i8, i8* %arg.addr, align 1
  %conv = uitofp i8 %0 to float
  ret float %conv
; CHECK-LABEL: @_Z7testfuch
; CHECK: mtvsrwz [[MOVEREG03:[0-9]+]], 3
; CHECK: xscvuxdsp 1, [[MOVEREG03]]
}

; Function Attrs: nounwind
define zeroext i8 @_Z7testucdd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptoui double %0 to i8
  ret i8 %conv
; CHECK-LABEL: @_Z7testucdd
; CHECK: xscvdpsxws [[CONVREG04:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG04]]
}

; Function Attrs: nounwind
define double @_Z7testduch(i8 zeroext %arg) {
entry:
  %arg.addr = alloca i8, align 1
  store i8 %arg, i8* %arg.addr, align 1
  %0 = load i8, i8* %arg.addr, align 1
  %conv = uitofp i8 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z7testduch
; CHECK: mtvsrwz [[MOVEREG04:[0-9]+]], 3
; CHECK: xscvuxddp 1, [[MOVEREG04]]
}

; Function Attrs: nounwind
define signext i16 @_Z6testsff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptosi float %0 to i16
  ret i16 %conv
; CHECK-LABEL: @_Z6testsff
; CHECK: xscvdpsxws [[CONVREG05:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG05]]
}

; Function Attrs: nounwind
define float @_Z6testfss(i16 signext %arg) {
entry:
  %arg.addr = alloca i16, align 2
  store i16 %arg, i16* %arg.addr, align 2
  %0 = load i16, i16* %arg.addr, align 2
  %conv = sitofp i16 %0 to float
  ret float %conv
; CHECK-LABEL: @_Z6testfss
; CHECK: mtvsrwa [[MOVEREG05:[0-9]+]], 3
; CHECK: xscvsxdsp 1, [[MOVEREG05]]
}

; Function Attrs: nounwind
define signext i16 @_Z6testsdd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptosi double %0 to i16
  ret i16 %conv
; CHECK-LABEL: @_Z6testsdd
; CHECK: xscvdpsxws [[CONVREG06:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG06]]
}

; Function Attrs: nounwind
define double @_Z6testdss(i16 signext %arg) {
entry:
  %arg.addr = alloca i16, align 2
  store i16 %arg, i16* %arg.addr, align 2
  %0 = load i16, i16* %arg.addr, align 2
  %conv = sitofp i16 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z6testdss
; CHECK: mtvsrwa [[MOVEREG06:[0-9]+]], 3
; CHECK: xscvsxddp 1, [[MOVEREG06]]
}

; Function Attrs: nounwind
define zeroext i16 @_Z7testusff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptoui float %0 to i16
  ret i16 %conv
; CHECK-LABEL: @_Z7testusff
; CHECK: xscvdpsxws [[CONVREG07:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG07]]
}

; Function Attrs: nounwind
define float @_Z7testfust(i16 zeroext %arg) {
entry:
  %arg.addr = alloca i16, align 2
  store i16 %arg, i16* %arg.addr, align 2
  %0 = load i16, i16* %arg.addr, align 2
  %conv = uitofp i16 %0 to float
  ret float %conv
; CHECK-LABEL: @_Z7testfust
; CHECK: mtvsrwz [[MOVEREG07:[0-9]+]], 3
; CHECK: xscvuxdsp 1, [[MOVEREG07]]
}

; Function Attrs: nounwind
define zeroext i16 @_Z7testusdd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptoui double %0 to i16
  ret i16 %conv
; CHECK-LABEL: @_Z7testusdd
; CHECK: xscvdpsxws [[CONVREG08:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG08]]
}

; Function Attrs: nounwind
define double @_Z7testdust(i16 zeroext %arg) {
entry:
  %arg.addr = alloca i16, align 2
  store i16 %arg, i16* %arg.addr, align 2
  %0 = load i16, i16* %arg.addr, align 2
  %conv = uitofp i16 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z7testdust
; CHECK: mtvsrwz [[MOVEREG08:[0-9]+]], 3
; CHECK: xscvuxddp 1, [[MOVEREG08]]
}

; Function Attrs: nounwind
define signext i32 @_Z6testiff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptosi float %0 to i32
  ret i32 %conv
; CHECK-LABEL: @_Z6testiff
; CHECK: xscvdpsxws [[CONVREG09:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG09]]
}

; Function Attrs: nounwind
define float @_Z6testfii(i32 signext %arg) {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  %0 = load i32, i32* %arg.addr, align 4
  %conv = sitofp i32 %0 to float
  ret float %conv
; CHECK-LABEL: @_Z6testfii
; CHECK: mtvsrwa [[MOVEREG09:[0-9]+]], 3
; CHECK: xscvsxdsp 1, [[MOVEREG09]]
}

; Function Attrs: nounwind
define signext i32 @_Z6testidd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptosi double %0 to i32
  ret i32 %conv
; CHECK-LABEL: @_Z6testidd
; CHECK: xscvdpsxws [[CONVREG10:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG10]]
}

; Function Attrs: nounwind
define double @_Z6testdii(i32 signext %arg) {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  %0 = load i32, i32* %arg.addr, align 4
  %conv = sitofp i32 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z6testdii
; CHECK: mtvsrwa [[MOVEREG10:[0-9]+]], 3
; CHECK: xscvsxddp 1, [[MOVEREG10]]
}

; Function Attrs: nounwind
define zeroext i32 @_Z7testuiff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptoui float %0 to i32
  ret i32 %conv
; CHECK-LABEL: @_Z7testuiff
; CHECK: xscvdpuxws [[CONVREG11:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG11]]
}

; Function Attrs: nounwind
define float @_Z7testfuij(i32 zeroext %arg) {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  %0 = load i32, i32* %arg.addr, align 4
  %conv = uitofp i32 %0 to float
  ret float %conv
; CHECK-LABEL: @_Z7testfuij
; CHECK: mtvsrwz [[MOVEREG11:[0-9]+]], 3
; CHECK: xscvuxdsp 1, [[MOVEREG11]]
}

; Function Attrs: nounwind
define zeroext i32 @_Z7testuidd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptoui double %0 to i32
  ret i32 %conv
; CHECK-LABEL: @_Z7testuidd
; CHECK: xscvdpuxws [[CONVREG12:[0-9]+]], 1
; CHECK: mfvsrwz 3, [[CONVREG12]]
}

; Function Attrs: nounwind
define double @_Z7testduij(i32 zeroext %arg) {
entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  %0 = load i32, i32* %arg.addr, align 4
  %conv = uitofp i32 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z7testduij
; CHECK: mtvsrwz [[MOVEREG12:[0-9]+]], 3
; CHECK: xscvuxddp 1, [[MOVEREG12]]
}

; Function Attrs: nounwind
define i64 @_Z7testllff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptosi float %0 to i64
  ret i64 %conv
; CHECK-LABEL: @_Z7testllff
; CHECK: xscvdpsxds [[CONVREG13:[0-9]+]], 1
; CHECK: mffprd 3, [[CONVREG13]]
}

; Function Attrs: nounwind
define float @_Z7testfllx(i64 %arg) {
entry:
  %arg.addr = alloca i64, align 8
  store i64 %arg, i64* %arg.addr, align 8
  %0 = load i64, i64* %arg.addr, align 8
  %conv = sitofp i64 %0 to float
  ret float %conv
; CHECK-LABEL:@_Z7testfllx
; CHECK: mtvsrd [[MOVEREG13:[0-9]+]], 3
; CHECK: xscvsxdsp 1, [[MOVEREG13]]
}

; Function Attrs: nounwind
define i64 @_Z7testlldd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptosi double %0 to i64
  ret i64 %conv
; CHECK-LABEL: @_Z7testlldd
; CHECK: xscvdpsxds [[CONVREG14:[0-9]+]], 1
; CHECK: mffprd 3, [[CONVREG14]]
}

; Function Attrs: nounwind
define double @_Z7testdllx(i64 %arg) {
entry:
  %arg.addr = alloca i64, align 8
  store i64 %arg, i64* %arg.addr, align 8
  %0 = load i64, i64* %arg.addr, align 8
  %conv = sitofp i64 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z7testdllx
; CHECK: mtvsrd [[MOVEREG14:[0-9]+]], 3
; CHECK: xscvsxddp 1, [[MOVEREG14]]
}

; Function Attrs: nounwind
define i64 @_Z8testullff(float %arg) {
entry:
  %arg.addr = alloca float, align 4
  store float %arg, float* %arg.addr, align 4
  %0 = load float, float* %arg.addr, align 4
  %conv = fptoui float %0 to i64
  ret i64 %conv
; CHECK-LABEL: @_Z8testullff
; CHECK: xscvdpuxds [[CONVREG15:[0-9]+]], 1
; CHECK: mffprd 3, [[CONVREG15]]
}

; Function Attrs: nounwind
define float @_Z8testfully(i64 %arg) {
entry:
  %arg.addr = alloca i64, align 8
  store i64 %arg, i64* %arg.addr, align 8
  %0 = load i64, i64* %arg.addr, align 8
  %conv = uitofp i64 %0 to float
  ret float %conv
; CHECK-LABEL: @_Z8testfully
; CHECK: mtvsrd [[MOVEREG15:[0-9]+]], 3
; CHECK: xscvuxdsp 1, [[MOVEREG15]]
}

; Function Attrs: nounwind
define i64 @_Z8testulldd(double %arg) {
entry:
  %arg.addr = alloca double, align 8
  store double %arg, double* %arg.addr, align 8
  %0 = load double, double* %arg.addr, align 8
  %conv = fptoui double %0 to i64
  ret i64 %conv
; CHECK-LABEL: @_Z8testulldd
; CHECK: xscvdpuxds [[CONVREG16:[0-9]+]], 1
; CHECK: mffprd 3, [[CONVREG16]]
}

; Function Attrs: nounwind
define double @_Z8testdully(i64 %arg) {
entry:
  %arg.addr = alloca i64, align 8
  store i64 %arg, i64* %arg.addr, align 8
  %0 = load i64, i64* %arg.addr, align 8
  %conv = uitofp i64 %0 to double
  ret double %conv
; CHECK-LABEL: @_Z8testdully
; CHECK: mtvsrd [[MOVEREG16:[0-9]+]], 3
; CHECK: xscvuxddp 1, [[MOVEREG16]]
}
