; RUN: llc < %s -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s
; RUN: llc < %s -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 | FileCheck %s -check-prefix=CHECK-LE

; The build[csilf] functions simply test the scalar_to_vector handling with
; direct moves. This corresponds to the "insertelement" instruction. Subsequent
; to this, there will be a splat corresponding to the shufflevector.

@d = common global double 0.000000e+00, align 8

; Function Attrs: nounwind
define <16 x i8> @buildc(i8 zeroext %a) {
entry:
  %a.addr = alloca i8, align 1
  store i8 %a, i8* %a.addr, align 1
  %0 = load i8, i8* %a.addr, align 1
  %splat.splatinsert = insertelement <16 x i8> undef, i8 %0, i32 0
  %splat.splat = shufflevector <16 x i8> %splat.splatinsert, <16 x i8> undef, <16 x i32> zeroinitializer
  ret <16 x i8> %splat.splat
; CHECK: sldi [[REG1:[0-9]+]], 3, 56
; CHECK: mtvsrd {{[0-9]+}}, [[REG1]]
; CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; CHECK-LE: xxswapd {{[0-9]+}}, [[REG1]]
}

; Function Attrs: nounwind
define <8 x i16> @builds(i16 zeroext %a) {
entry:
  %a.addr = alloca i16, align 2
  store i16 %a, i16* %a.addr, align 2
  %0 = load i16, i16* %a.addr, align 2
  %splat.splatinsert = insertelement <8 x i16> undef, i16 %0, i32 0
  %splat.splat = shufflevector <8 x i16> %splat.splatinsert, <8 x i16> undef, <8 x i32> zeroinitializer
  ret <8 x i16> %splat.splat
; CHECK: sldi [[REG1:[0-9]+]], 3, 48
; CHECK: mtvsrd {{[0-9]+}}, [[REG1]]
; CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; CHECK-LE: xxswapd {{[0-9]+}}, [[REG1]]
}

; Function Attrs: nounwind
define <4 x i32> @buildi(i32 zeroext %a) {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, i32* %a.addr, align 4
  %0 = load i32, i32* %a.addr, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; CHECK: sldi [[REG1:[0-9]+]], 3, 32
; CHECK: mtvsrd {{[0-9]+}}, [[REG1]]
; CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; CHECK-LE: xxswapd {{[0-9]+}}, [[REG1]]
}

; Function Attrs: nounwind
define <2 x i64> @buildl(i64 %a) {
entry:
  %a.addr = alloca i64, align 8
  store i64 %a, i64* %a.addr, align 8
  %0 = load i64, i64* %a.addr, align 8
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %0, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; CHECK: mtvsrd {{[0-9]+}}, 3
; CHECK-LE: mtvsrd [[REG1:[0-9]+]], 3
; CHECK-LE: xxspltd [[REG1]], [[REG1]], 0
}

; Function Attrs: nounwind
define <4 x float> @buildf(float %a) {
entry:
  %a.addr = alloca float, align 4
  store float %a, float* %a.addr, align 4
  %0 = load float, float* %a.addr, align 4
  %splat.splatinsert = insertelement <4 x float> undef, float %0, i32 0
  %splat.splat = shufflevector <4 x float> %splat.splatinsert, <4 x float> undef, <4 x i32> zeroinitializer
  ret <4 x float> %splat.splat
; CHECK: xscvdpspn {{[0-9]+}}, 1
; CHECK-LE: xscvdpspn [[REG1:[0-9]+]], 1
; CHECK-LE: xxsldwi {{[0-9]+}}, [[REG1]], [[REG1]], 1
}

; The optimization to remove stack operations from PPCDAGToDAGISel::Select
; should still trigger for v2f64, producing an lxvdsx.
; Function Attrs: nounwind
define <2 x double> @buildd() #0 {
entry:
  %0 = load double, double* @d, align 8
  %splat.splatinsert = insertelement <2 x double> undef, double %0, i32 0
  %splat.splat = shufflevector <2 x double> %splat.splatinsert, <2 x double> undef, <2 x i32> zeroinitializer
  ret <2 x double> %splat.splat
; CHECK: ld [[REG1:[0-9]+]], .LC0@toc@l
; CHECK: lxvdsx 34, 0, [[REG1]]
; CHECK-LE: ld [[REG1:[0-9]+]], .LC0@toc@l
; CHECK-LE: lxvdsx 34, 0, [[REG1]]
}

; Function Attrs: nounwind
define signext i8 @getsc0(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 0
  ret i8 %vecext
; CHECK-LABEL: @getsc0
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 8, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc0
; CHECK-LE: mfvsrd 3,
; CHECK-LE: clrldi 3, 3, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc1(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 1
  ret i8 %vecext
; CHECK-LABEL: @getsc1
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 16, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc1
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 56, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc2(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 2
  ret i8 %vecext
; CHECK-LABEL: @getsc2
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 24, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc2
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 48, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc3(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 3
  ret i8 %vecext
; CHECK-LABEL: @getsc3
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 32, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc3
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 40, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc4(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 4
  ret i8 %vecext
; CHECK-LABEL: @getsc4
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 40, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc4
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 32, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc5(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 5
  ret i8 %vecext
; CHECK-LABEL: @getsc5
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 48, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc5
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 24, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc6(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 6
  ret i8 %vecext
; CHECK-LABEL: @getsc6
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 56, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc6
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 16, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc7(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 7
  ret i8 %vecext
; CHECK-LABEL: @getsc7
; CHECK: mfvsrd 3, 34
; CHECK: clrldi 3, 3, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc7
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 8, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc8(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 8
  ret i8 %vecext
; CHECK-LABEL: @getsc8
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 8, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc8
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: clrldi 3, 3, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc9(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 9
  ret i8 %vecext
; CHECK-LABEL: @getsc9
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 16, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc9
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 56, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc10(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 10
  ret i8 %vecext
; CHECK-LABEL: @getsc10
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 24, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc10
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 48, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc11(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 11
  ret i8 %vecext
; CHECK-LABEL: @getsc11
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 32, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc11
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 40, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc12(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 12
  ret i8 %vecext
; CHECK-LABEL: @getsc12
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 40, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc12
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 32, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc13(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 13
  ret i8 %vecext
; CHECK-LABEL: @getsc13
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 48, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc13
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 24, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc14(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 14
  ret i8 %vecext
; CHECK-LABEL: @getsc14
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 56, 56
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc14
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 16, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define signext i8 @getsc15(<16 x i8> %vsc) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 15
  ret i8 %vecext
; CHECK-LABEL: @getsc15
; CHECK: mfvsrd 3,
; CHECK: extsb 3, 3
; CHECK-LE-LABEL: @getsc15
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 8, 56
; CHECK-LE: extsb 3, 3
}

; Function Attrs: nounwind
define zeroext i8 @getuc0(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 0
  ret i8 %vecext
; CHECK-LABEL: @getuc0
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 8, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc0
; CHECK-LE: mfvsrd 3,
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc1(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 1
  ret i8 %vecext
; CHECK-LABEL: @getuc1
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 16, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc1
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 56, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc2(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 2
  ret i8 %vecext
; CHECK-LABEL: @getuc2
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 24, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc2
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 48, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc3(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 3
  ret i8 %vecext
; CHECK-LABEL: @getuc3
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 32, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc3
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 40, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc4(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 4
  ret i8 %vecext
; CHECK-LABEL: @getuc4
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 40, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc4
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 32, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc5(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 5
  ret i8 %vecext
; CHECK-LABEL: @getuc5
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 48, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc5
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 24, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc6(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 6
  ret i8 %vecext
; CHECK-LABEL: @getuc6
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 56, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc6
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 16, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc7(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 7
  ret i8 %vecext
; CHECK-LABEL: @getuc7
; CHECK: mfvsrd 3, 34
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc7
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 8, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc8(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 8
  ret i8 %vecext
; CHECK-LABEL: @getuc8
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 8, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc8
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc9(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 9
  ret i8 %vecext
; CHECK-LABEL: @getuc9
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 16, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc9
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 56, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc10(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 10
  ret i8 %vecext
; CHECK-LABEL: @getuc10
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 24, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc10
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 48, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc11(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 11
  ret i8 %vecext
; CHECK-LABEL: @getuc11
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 32, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc11
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 40, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc12(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 12
  ret i8 %vecext
; CHECK-LABEL: @getuc12
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 40, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc12
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 32, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc13(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 13
  ret i8 %vecext
; CHECK-LABEL: @getuc13
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 48, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc13
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 24, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc14(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 14
  ret i8 %vecext
; CHECK-LABEL: @getuc14
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 56, 56
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc14
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 16, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define zeroext i8 @getuc15(<16 x i8> %vuc) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %vecext = extractelement <16 x i8> %0, i32 15
  ret i8 %vecext
; CHECK-LABEL: @getuc15
; CHECK: mfvsrd 3,
; CHECK: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getuc15
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 8, 56
; CHECK-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define signext i8 @getvelsc(<16 x i8> %vsc, i32 signext %i) {
entry:
  %vsc.addr = alloca <16 x i8>, align 16
  %i.addr = alloca i32, align 4
  store <16 x i8> %vsc, <16 x i8>* %vsc.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <16 x i8>, <16 x i8>* %vsc.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <16 x i8> %0, i32 %1
  ret i8 %vecext
; CHECK-LABEL: @getvelsc
; CHECK-DAG: andi. [[ANDI:[0-9]+]], {{[0-9]+}}, 8
; CHECK-DAG: lvsl [[SHMSK:[0-9]+]], 0, [[ANDI]]
; CHECK-DAG: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG: li [[IMM7:[0-9]+]], 7
; CHECK-DAG: andc [[ANDC:[0-9]+]], [[IMM7]]
; CHECK-DAG: sldi [[SHL:[0-9]+]], [[ANDC]], 3
; CHECK-DAG: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG: extsb 3, 3
; CHECK-LE-LABEL: @getvelsc
; CHECK-DAG-LE: li [[IMM8:[0-9]+]], 8
; CHECK-DAG-LE: andc [[ANDC:[0-9]+]], [[IMM8]]
; CHECK-DAG-LE: lvsl [[SHMSK:[0-9]+]], 0, [[ANDC]]
; CHECK-DAG-LE: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG-LE: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG-LE: li [[IMM7:[0-9]+]], 7
; CHECK-DAG-LE: and [[AND:[0-9]+]], [[IMM7]]
; CHECK-DAG-LE: sldi [[SHL:[0-9]+]], [[AND]], 3
; CHECK-DAG-LE: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG-LE: extsb 3, 3
}

; Function Attrs: nounwind
define zeroext i8 @getveluc(<16 x i8> %vuc, i32 signext %i) {
entry:
  %vuc.addr = alloca <16 x i8>, align 16
  %i.addr = alloca i32, align 4
  store <16 x i8> %vuc, <16 x i8>* %vuc.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <16 x i8>, <16 x i8>* %vuc.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <16 x i8> %0, i32 %1
  ret i8 %vecext
; CHECK-LABEL: @getveluc
; CHECK-DAG: andi. [[ANDI:[0-9]+]], {{[0-9]+}}, 8
; CHECK-DAG: lvsl [[SHMSK:[0-9]+]], 0, [[ANDI]]
; CHECK-DAG: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG: li [[IMM7:[0-9]+]], 7
; CHECK-DAG: andc [[ANDC:[0-9]+]], [[IMM7]]
; CHECK-DAG: sldi [[SHL:[0-9]+]], [[ANDC]], 3
; CHECK-DAG: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG: clrldi   3, 3, 56
; CHECK-LE-LABEL: @getveluc
; CHECK-DAG-LE: li [[IMM8:[0-9]+]], 8
; CHECK-DAG-LE: andc [[ANDC:[0-9]+]], [[IMM8]]
; CHECK-DAG-LE: lvsl [[SHMSK:[0-9]+]], 0, [[ANDC]]
; CHECK-DAG-LE: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG-LE: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG-LE: li [[IMM7:[0-9]+]], 7
; CHECK-DAG-LE: and [[AND:[0-9]+]], [[IMM7]]
; CHECK-DAG-LE: sldi [[SHL:[0-9]+]], [[AND]], 3
; CHECK-DAG-LE: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG-LE: clrldi   3, 3, 56
}

; Function Attrs: nounwind
define signext i16 @getss0(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 0
  ret i16 %vecext
; CHECK-LABEL: @getss0
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 16, 48
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss0
; CHECK-LE: mfvsrd 3,
; CHECK-LE: clrldi 3, 3, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define signext i16 @getss1(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 1
  ret i16 %vecext
; CHECK-LABEL: @getss1
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 32, 48
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss1
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 48, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define signext i16 @getss2(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 2
  ret i16 %vecext
; CHECK-LABEL: @getss2
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 48, 48
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss2
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 32, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define signext i16 @getss3(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 3
  ret i16 %vecext
; CHECK-LABEL: @getss3
; CHECK: mfvsrd 3, 34
; CHECK: clrldi 3, 3, 48
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss3
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 16, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define signext i16 @getss4(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 4
  ret i16 %vecext
; CHECK-LABEL: @getss4
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 16, 48
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss4
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: clrldi 3, 3, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define signext i16 @getss5(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 5
  ret i16 %vecext
; CHECK-LABEL: @getss5
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 32, 48
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss5
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 48, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define signext i16 @getss6(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 6
  ret i16 %vecext
; CHECK-LABEL: @getss6
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 48, 48
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss6
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 32, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define signext i16 @getss7(<8 x i16> %vss) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 7
  ret i16 %vecext
; CHECK-LABEL: @getss7
; CHECK: mfvsrd 3,
; CHECK: extsh 3, 3
; CHECK-LE-LABEL: @getss7
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 16, 48
; CHECK-LE: extsh 3, 3
}

; Function Attrs: nounwind
define zeroext i16 @getus0(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 0
  ret i16 %vecext
; CHECK-LABEL: @getus0
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 16, 48
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus0
; CHECK-LE: mfvsrd 3,
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define zeroext i16 @getus1(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 1
  ret i16 %vecext
; CHECK-LABEL: @getus1
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 32, 48
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus1
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 48, 48
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define zeroext i16 @getus2(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 2
  ret i16 %vecext
; CHECK-LABEL: @getus2
; CHECK: mfvsrd 3, 34
; CHECK: rldicl 3, 3, 48, 48
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus2
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 32, 48
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define zeroext i16 @getus3(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 3
  ret i16 %vecext
; CHECK-LABEL: @getus3
; CHECK: mfvsrd 3, 34
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus3
; CHECK-LE: mfvsrd 3,
; CHECK-LE: rldicl 3, 3, 16, 48
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define zeroext i16 @getus4(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 4
  ret i16 %vecext
; CHECK-LABEL: @getus4
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 16, 48
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus4
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define zeroext i16 @getus5(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 5
  ret i16 %vecext
; CHECK-LABEL: @getus5
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 32, 48
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus5
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 48, 48
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define zeroext i16 @getus6(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 6
  ret i16 %vecext
; CHECK-LABEL: @getus6
; CHECK: mfvsrd 3,
; CHECK: rldicl 3, 3, 48, 48
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus6
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 32, 48
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define zeroext i16 @getus7(<8 x i16> %vus) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %vecext = extractelement <8 x i16> %0, i32 7
  ret i16 %vecext
; CHECK-LABEL: @getus7
; CHECK: mfvsrd 3,
; CHECK: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getus7
; CHECK-LE: mfvsrd 3, 34
; CHECK-LE: rldicl 3, 3, 16, 48
; CHECK-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define signext i16 @getvelss(<8 x i16> %vss, i32 signext %i) {
entry:
  %vss.addr = alloca <8 x i16>, align 16
  %i.addr = alloca i32, align 4
  store <8 x i16> %vss, <8 x i16>* %vss.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <8 x i16>, <8 x i16>* %vss.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <8 x i16> %0, i32 %1
  ret i16 %vecext
; CHECK-LABEL: @getvelss
; CHECK-DAG: andi. [[ANDI:[0-9]+]], {{[0-9]+}}, 4
; CHECK-DAG: sldi [[MUL2:[0-9]+]], [[ANDI]], 1
; CHECK-DAG: lvsl [[SHMSK:[0-9]+]], 0, [[MUL2]]
; CHECK-DAG: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG: li [[IMM3:[0-9]+]], 3
; CHECK-DAG: andc [[ANDC:[0-9]+]], [[IMM3]]
; CHECK-DAG: sldi [[SHL:[0-9]+]], [[ANDC]], 4
; CHECK-DAG: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG: extsh 3, 3
; CHECK-LE-LABEL: @getvelss
; CHECK-DAG-LE: li [[IMM4:[0-9]+]], 4
; CHECK-DAG-LE: andc [[ANDC:[0-9]+]], [[IMM4]]
; CHECK-DAG-LE: sldi [[MUL2:[0-9]+]], [[ANDC]], 1
; CHECK-DAG-LE: lvsl [[SHMSK:[0-9]+]], 0, [[MUL2]]
; CHECK-DAG-LE: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG-LE: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG-LE: li [[IMM3:[0-9]+]], 3
; CHECK-DAG-LE: and [[AND:[0-9]+]], [[IMM3]]
; CHECK-DAG-LE: sldi [[SHL:[0-9]+]], [[AND]], 4
; CHECK-DAG-LE: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG-LE: extsh 3, 3
}

; Function Attrs: nounwind
define zeroext i16 @getvelus(<8 x i16> %vus, i32 signext %i) {
entry:
  %vus.addr = alloca <8 x i16>, align 16
  %i.addr = alloca i32, align 4
  store <8 x i16> %vus, <8 x i16>* %vus.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <8 x i16>, <8 x i16>* %vus.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <8 x i16> %0, i32 %1
  ret i16 %vecext
; CHECK-LABEL: @getvelus
; CHECK-DAG: andi. [[ANDI:[0-9]+]], {{[0-9]+}}, 4
; CHECK-DAG: sldi [[MUL2:[0-9]+]], [[ANDI]], 1
; CHECK-DAG: lvsl [[SHMSK:[0-9]+]], 0, [[MUL2]]
; CHECK-DAG: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG: li [[IMM3:[0-9]+]], 3
; CHECK-DAG: andc [[ANDC:[0-9]+]], [[IMM3]]
; CHECK-DAG: sldi [[SHL:[0-9]+]], [[ANDC]], 4
; CHECK-DAG: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG: clrldi   3, 3, 48
; CHECK-LE-LABEL: @getvelus
; CHECK-DAG-LE: li [[IMM4:[0-9]+]], 4
; CHECK-DAG-LE: andc [[ANDC:[0-9]+]], [[IMM4]]
; CHECK-DAG-LE: sldi [[MUL2:[0-9]+]], [[ANDC]], 1
; CHECK-DAG-LE: lvsl [[SHMSK:[0-9]+]], 0, [[MUL2]]
; CHECK-DAG-LE: vperm [[PERMD:[0-9]+]], {{[0-9]+}}, {{[0-9]+}}, [[SHMSK]]
; CHECK-DAG-LE: mfvsrd [[MOV:[0-9]+]],
; CHECK-DAG-LE: li [[IMM3:[0-9]+]], 3
; CHECK-DAG-LE: and [[AND:[0-9]+]], [[IMM3]]
; CHECK-DAG-LE: sldi [[SHL:[0-9]+]], [[AND]], 4
; CHECK-DAG-LE: srd 3, [[MOV]], [[SHL]]
; CHECK-DAG-LE: clrldi   3, 3, 48
}

; Function Attrs: nounwind
define signext i32 @getsi0(<4 x i32> %vsi) {
entry:
  %vsi.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vsi, <4 x i32>* %vsi.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vsi.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 0
  ret i32 %vecext
; CHECK-LABEL: @getsi0
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 3
; CHECK: mfvsrwz 3, [[SHL]]
; CHECK: extsw 3, 3
; CHECK-LE-LABEL: @getsi0
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 2
; CHECK-LE: mfvsrwz 3, [[SHL]]
; CHECK-LE: extsw 3, 3
}

; Function Attrs: nounwind
define signext i32 @getsi1(<4 x i32> %vsi) {
entry:
  %vsi.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vsi, <4 x i32>* %vsi.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vsi.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 1
  ret i32 %vecext
; CHECK-LABEL: @getsi1
; CHECK: mfvsrwz 3, 34
; CHECK: extsw 3, 3
; CHECK-LE-LABEL: @getsi1
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 1
; CHECK-LE: mfvsrwz 3, [[SHL]]
; CHECK-LE: extsw 3, 3
}

; Function Attrs: nounwind
define signext i32 @getsi2(<4 x i32> %vsi) {
entry:
  %vsi.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vsi, <4 x i32>* %vsi.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vsi.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 2
  ret i32 %vecext
; CHECK-LABEL: @getsi2
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 1
; CHECK: mfvsrwz 3, [[SHL]]
; CHECK: extsw 3, 3
; CHECK-LE-LABEL: @getsi2
; CHECK-LE: mfvsrwz 3, 34
; CHECK-LE: extsw 3, 3
}

; Function Attrs: nounwind
define signext i32 @getsi3(<4 x i32> %vsi) {
entry:
  %vsi.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vsi, <4 x i32>* %vsi.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vsi.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 3
  ret i32 %vecext
; CHECK-LABEL: @getsi3
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 2
; CHECK: mfvsrwz 3, [[SHL]]
; CHECK: extsw 3, 3
; CHECK-LE-LABEL: @getsi3
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 3
; CHECK-LE: mfvsrwz 3, [[SHL]]
; CHECK-LE: extsw 3, 3
}

; Function Attrs: nounwind
define zeroext i32 @getui0(<4 x i32> %vui) {
entry:
  %vui.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vui, <4 x i32>* %vui.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vui.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 0
  ret i32 %vecext
; CHECK-LABEL: @getui0
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 3
; CHECK: mfvsrwz 3, [[SHL]]
; CHECK: clrldi   3, 3, 32
; CHECK-LE-LABEL: @getui0
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 2
; CHECK-LE: mfvsrwz 3, [[SHL]]
; CHECK-LE: clrldi   3, 3, 32
}

; Function Attrs: nounwind
define zeroext i32 @getui1(<4 x i32> %vui) {
entry:
  %vui.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vui, <4 x i32>* %vui.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vui.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 1
  ret i32 %vecext
; CHECK-LABEL: @getui1
; CHECK: mfvsrwz 3, 34
; CHECK: clrldi   3, 3, 32
; CHECK-LE-LABEL: @getui1
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 1
; CHECK-LE: mfvsrwz 3, [[SHL]]
; CHECK-LE: clrldi   3, 3, 32
}

; Function Attrs: nounwind
define zeroext i32 @getui2(<4 x i32> %vui) {
entry:
  %vui.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vui, <4 x i32>* %vui.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vui.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 2
  ret i32 %vecext
; CHECK-LABEL: @getui2
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 1
; CHECK: mfvsrwz 3, [[SHL]]
; CHECK: clrldi   3, 3, 32
; CHECK-LE-LABEL: @getui2
; CHECK-LE: mfvsrwz 3, 34
; CHECK-LE: clrldi   3, 3, 32
}

; Function Attrs: nounwind
define zeroext i32 @getui3(<4 x i32> %vui) {
entry:
  %vui.addr = alloca <4 x i32>, align 16
  store <4 x i32> %vui, <4 x i32>* %vui.addr, align 16
  %0 = load <4 x i32>, <4 x i32>* %vui.addr, align 16
  %vecext = extractelement <4 x i32> %0, i32 3
  ret i32 %vecext
; CHECK-LABEL: @getui3
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 2
; CHECK: mfvsrwz 3, [[SHL]]
; CHECK: clrldi   3, 3, 32
; CHECK-LE-LABEL: @getui3
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 3
; CHECK-LE: mfvsrwz 3, [[SHL]]
; CHECK-LE: clrldi   3, 3, 32
}

; Function Attrs: nounwind
define signext i32 @getvelsi(<4 x i32> %vsi, i32 signext %i) {
entry:
  %vsi.addr = alloca <4 x i32>, align 16
  %i.addr = alloca i32, align 4
  store <4 x i32> %vsi, <4 x i32>* %vsi.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <4 x i32>, <4 x i32>* %vsi.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <4 x i32> %0, i32 %1
  ret i32 %vecext
; CHECK-LABEL: @getvelsi
; CHECK-LE-LABEL: @getvelsi
; FIXME: add check patterns when variable element extraction is implemented
}

; Function Attrs: nounwind
define zeroext i32 @getvelui(<4 x i32> %vui, i32 signext %i) {
entry:
  %vui.addr = alloca <4 x i32>, align 16
  %i.addr = alloca i32, align 4
  store <4 x i32> %vui, <4 x i32>* %vui.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <4 x i32>, <4 x i32>* %vui.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <4 x i32> %0, i32 %1
  ret i32 %vecext
; CHECK-LABEL: @getvelui
; CHECK-LE-LABEL: @getvelui
; FIXME: add check patterns when variable element extraction is implemented
}

; Function Attrs: nounwind
define i64 @getsl0(<2 x i64> %vsl) {
entry:
  %vsl.addr = alloca <2 x i64>, align 16
  store <2 x i64> %vsl, <2 x i64>* %vsl.addr, align 16
  %0 = load <2 x i64>, <2 x i64>* %vsl.addr, align 16
  %vecext = extractelement <2 x i64> %0, i32 0
  ret i64 %vecext
; CHECK-LABEL: @getsl0
; CHECK: mfvsrd 3, 34
; CHECK-LE-LABEL: @getsl0
; CHECK-LE: xxswapd  [[SWP:[0-9]+]], 34
; CHECK-LE: mfvsrd 3, [[SWP]]
}

; Function Attrs: nounwind
define i64 @getsl1(<2 x i64> %vsl) {
entry:
  %vsl.addr = alloca <2 x i64>, align 16
  store <2 x i64> %vsl, <2 x i64>* %vsl.addr, align 16
  %0 = load <2 x i64>, <2 x i64>* %vsl.addr, align 16
  %vecext = extractelement <2 x i64> %0, i32 1
  ret i64 %vecext
; CHECK-LABEL: @getsl1
; CHECK: xxswapd  [[SWP:[0-9]+]], 34
; CHECK: mfvsrd 3, [[SWP]]
; CHECK-LE-LABEL: @getsl1
; CHECK-LE: mfvsrd 3, 34
}

; Function Attrs: nounwind
define i64 @getul0(<2 x i64> %vul) {
entry:
  %vul.addr = alloca <2 x i64>, align 16
  store <2 x i64> %vul, <2 x i64>* %vul.addr, align 16
  %0 = load <2 x i64>, <2 x i64>* %vul.addr, align 16
  %vecext = extractelement <2 x i64> %0, i32 0
  ret i64 %vecext
; CHECK-LABEL: @getul0
; CHECK: mfvsrd 3, 34
; CHECK-LE-LABEL: @getul0
; CHECK-LE: xxswapd  [[SWP:[0-9]+]], 34
; CHECK-LE: mfvsrd 3, [[SWP]]
}

; Function Attrs: nounwind
define i64 @getul1(<2 x i64> %vul) {
entry:
  %vul.addr = alloca <2 x i64>, align 16
  store <2 x i64> %vul, <2 x i64>* %vul.addr, align 16
  %0 = load <2 x i64>, <2 x i64>* %vul.addr, align 16
  %vecext = extractelement <2 x i64> %0, i32 1
  ret i64 %vecext
; CHECK-LABEL: @getul1
; CHECK: xxswapd  [[SWP:[0-9]+]], 34
; CHECK: mfvsrd 3, [[SWP]]
; CHECK-LE-LABEL: @getul1
; CHECK-LE: mfvsrd 3, 34
}

; Function Attrs: nounwind
define i64 @getvelsl(<2 x i64> %vsl, i32 signext %i) {
entry:
  %vsl.addr = alloca <2 x i64>, align 16
  %i.addr = alloca i32, align 4
  store <2 x i64> %vsl, <2 x i64>* %vsl.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <2 x i64>, <2 x i64>* %vsl.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <2 x i64> %0, i32 %1
  ret i64 %vecext
; CHECK-LABEL: @getvelsl
; CHECK-LE-LABEL: @getvelsl
; FIXME: add check patterns when variable element extraction is implemented
}

; Function Attrs: nounwind
define i64 @getvelul(<2 x i64> %vul, i32 signext %i) {
entry:
  %vul.addr = alloca <2 x i64>, align 16
  %i.addr = alloca i32, align 4
  store <2 x i64> %vul, <2 x i64>* %vul.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <2 x i64>, <2 x i64>* %vul.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <2 x i64> %0, i32 %1
  ret i64 %vecext
; CHECK-LABEL: @getvelul
; CHECK-LE-LABEL: @getvelul
; FIXME: add check patterns when variable element extraction is implemented
}

; Function Attrs: nounwind
define float @getf0(<4 x float> %vf) {
entry:
  %vf.addr = alloca <4 x float>, align 16
  store <4 x float> %vf, <4 x float>* %vf.addr, align 16
  %0 = load <4 x float>, <4 x float>* %vf.addr, align 16
  %vecext = extractelement <4 x float> %0, i32 0
  ret float %vecext
; CHECK-LABEL: @getf0
; CHECK: xscvspdpn 1, 34
; CHECK-LE-LABEL: @getf0
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 3
; CHECK-LE: xscvspdpn 1, [[SHL]]
}

; Function Attrs: nounwind
define float @getf1(<4 x float> %vf) {
entry:
  %vf.addr = alloca <4 x float>, align 16
  store <4 x float> %vf, <4 x float>* %vf.addr, align 16
  %0 = load <4 x float>, <4 x float>* %vf.addr, align 16
  %vecext = extractelement <4 x float> %0, i32 1
  ret float %vecext
; CHECK-LABEL: @getf1
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 1
; CHECK: xscvspdpn 1, [[SHL]]
; CHECK-LE-LABEL: @getf1
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 2
; CHECK-LE: xscvspdpn 1, [[SHL]]
}

; Function Attrs: nounwind
define float @getf2(<4 x float> %vf) {
entry:
  %vf.addr = alloca <4 x float>, align 16
  store <4 x float> %vf, <4 x float>* %vf.addr, align 16
  %0 = load <4 x float>, <4 x float>* %vf.addr, align 16
  %vecext = extractelement <4 x float> %0, i32 2
  ret float %vecext
; CHECK-LABEL: @getf2
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 2
; CHECK: xscvspdpn 1, [[SHL]]
; CHECK-LE-LABEL: @getf2
; CHECK-LE: xxsldwi [[SHL:[0-9]+]], 34, 34, 1
; CHECK-LE: xscvspdpn 1, [[SHL]]
}

; Function Attrs: nounwind
define float @getf3(<4 x float> %vf) {
entry:
  %vf.addr = alloca <4 x float>, align 16
  store <4 x float> %vf, <4 x float>* %vf.addr, align 16
  %0 = load <4 x float>, <4 x float>* %vf.addr, align 16
  %vecext = extractelement <4 x float> %0, i32 3
  ret float %vecext
; CHECK-LABEL: @getf3
; CHECK: xxsldwi [[SHL:[0-9]+]], 34, 34, 3
; CHECK: xscvspdpn 1, [[SHL]]
; CHECK-LE-LABEL: @getf3
; CHECK-LE: xscvspdpn 1, 34
}

; Function Attrs: nounwind
define float @getvelf(<4 x float> %vf, i32 signext %i) {
entry:
  %vf.addr = alloca <4 x float>, align 16
  %i.addr = alloca i32, align 4
  store <4 x float> %vf, <4 x float>* %vf.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <4 x float>, <4 x float>* %vf.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <4 x float> %0, i32 %1
  ret float %vecext
; CHECK-LABEL: @getvelf
; CHECK-LE-LABEL: @getvelf
; FIXME: add check patterns when variable element extraction is implemented
}

; Function Attrs: nounwind
define double @getd0(<2 x double> %vd) {
entry:
  %vd.addr = alloca <2 x double>, align 16
  store <2 x double> %vd, <2 x double>* %vd.addr, align 16
  %0 = load <2 x double>, <2 x double>* %vd.addr, align 16
  %vecext = extractelement <2 x double> %0, i32 0
  ret double %vecext
; CHECK-LABEL: @getd0
; CHECK: xxlor 1, 34, 34
; CHECK-LE-LABEL: @getd0
; CHECK-LE: xxswapd  1, 34
}

; Function Attrs: nounwind
define double @getd1(<2 x double> %vd) {
entry:
  %vd.addr = alloca <2 x double>, align 16
  store <2 x double> %vd, <2 x double>* %vd.addr, align 16
  %0 = load <2 x double>, <2 x double>* %vd.addr, align 16
  %vecext = extractelement <2 x double> %0, i32 1
  ret double %vecext
; CHECK-LABEL: @getd1
; CHECK: xxswapd  1, 34
; CHECK-LE-LABEL: @getd1
; CHECK-LE: xxlor 1, 34, 34
}

; Function Attrs: nounwind
define double @getveld(<2 x double> %vd, i32 signext %i) {
entry:
  %vd.addr = alloca <2 x double>, align 16
  %i.addr = alloca i32, align 4
  store <2 x double> %vd, <2 x double>* %vd.addr, align 16
  store i32 %i, i32* %i.addr, align 4
  %0 = load <2 x double>, <2 x double>* %vd.addr, align 16
  %1 = load i32, i32* %i.addr, align 4
  %vecext = extractelement <2 x double> %0, i32 %1
  ret double %vecext
; CHECK-LABEL: @getveld
; CHECK-LE-LABEL: @getveld
; FIXME: add check patterns when variable element extraction is implemented
}
