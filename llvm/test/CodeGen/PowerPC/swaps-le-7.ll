; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu -O3 < %s | FileCheck %s
;
; This is a regression test based on https://llvm.org/bugs/show_bug.cgi?id=27735
;

@G1 = global <2 x double> <double 2.0, double -10.0>
@G2 = global <2 x double> <double 3.0, double 4.0>
@G3 = global <2 x double> <double 5.0, double 6.0>
@G4 = global <2 x double> <double 7.0, double 8.0>

; CHECK-LABEL: @zg
; CHECK: lxvdsx
; CHECK-NEXT: lxvdsx
; CHECK-NEXT: xvmuldp
; CHECK-DAG: xvmuldp
; CHECK-DAG: xvsubdp
; CHECK-DAG: xvadddp
; CHECK-DAG: xxswapd
; CHECK-DAG: xxpermdi
; CHECK-DAG: xvsubdp
; CHECK: xxswapd
; CHECK-NEXT: stxvd2x
; CHECK: blr

; Function Attrs: noinline
define void @zg(i8* %.G0011_640.0, i8* %.G0012_642.0, <2 x double>* %JJ, <2 x double>* %.ka0000_391, double %.unpack, double %.unpack66) #0 {
L.JA291:
  %Z.L.JA291.2 = load <2 x double>, <2 x double>* %.ka0000_391, align 16
  store <2 x double> %Z.L.JA291.2, <2 x double>* %JJ, align 8
  %Z.L.JA291.3 = bitcast i8* %.G0012_642.0 to <2 x double>*
  %Z.L.JA291.4 = load <2 x double>, <2 x double>* %Z.L.JA291.3, align 1
  %.elt136 = bitcast i8* %.G0011_640.0 to double*
  %.unpack137 = load double, double* %.elt136, align 1
  %.elt138 = getelementptr inbounds i8, i8* %.G0011_640.0, i64 8
  %Z.L.JA291.5 = bitcast i8* %.elt138 to double*
  %.unpack139 = load double, double* %Z.L.JA291.5, align 1
  %Z.L.JA291.6 = insertelement <2 x double> undef, double %.unpack137, i32 0
  %Z.L.JA291.7 = insertelement <2 x double> %Z.L.JA291.6, double %.unpack137, i32 1
  %Z.L.JA291.8 = fmul <2 x double> %Z.L.JA291.2, %Z.L.JA291.7
  %Z.L.JA291.9 = shufflevector <2 x double> %Z.L.JA291.2, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  %Z.L.JA291.10 = insertelement <2 x double> undef, double %.unpack139, i32 0
  %Z.L.JA291.11 = insertelement <2 x double> %Z.L.JA291.10, double %.unpack139, i32 1
  %Z.L.JA291.12 = fmul <2 x double> %Z.L.JA291.9, %Z.L.JA291.11
  %Z.L.JA291.13 = fsub <2 x double> %Z.L.JA291.8, %Z.L.JA291.12
  %Z.L.JA291.14 = fadd <2 x double> %Z.L.JA291.8, %Z.L.JA291.12
  %Z.L.JA291.15 = shufflevector <2 x double> %Z.L.JA291.13, <2 x double> %Z.L.JA291.14, <2 x i32> <i32 0, i32 3>
  %Z.L.JA291.16 = fsub <2 x double> %Z.L.JA291.4, %Z.L.JA291.15
  %Z.L.JA291.17 = bitcast i8* %.G0012_642.0 to <2 x double>*
  store <2 x double> %Z.L.JA291.16, <2 x double>* %Z.L.JA291.17, align 8
  %.. = bitcast <2 x double>* %JJ to i32*
  %.pre = load i32, i32* %.., align 32
  ret void
}

attributes #0 = { noinline }
