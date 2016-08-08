; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

@vda = common global <2 x double> zeroinitializer, align 16
@vdb = common global <2 x double> zeroinitializer, align 16
@vdr = common global <2 x double> zeroinitializer, align 16
@vfa = common global <4 x float> zeroinitializer, align 16
@vfb = common global <4 x float> zeroinitializer, align 16
@vfr = common global <4 x float> zeroinitializer, align 16
@vbllr = common global <2 x i64> zeroinitializer, align 16
@vbir = common global <4 x i32> zeroinitializer, align 16
@vblla = common global <2 x i64> zeroinitializer, align 16
@vbllb = common global <2 x i64> zeroinitializer, align 16
@vbia = common global <4 x i32> zeroinitializer, align 16
@vbib = common global <4 x i32> zeroinitializer, align 16

; Function Attrs: nounwind
define void @test1() {
entry:
  %0 = load <2 x double>, <2 x double>* @vda, align 16
  %1 = load <2 x double>, <2 x double>* @vdb, align 16
  %2 = call <2 x double> @llvm.ppc.vsx.xvdivdp(<2 x double> %0, <2 x double> %1)
  store <2 x double> %2, <2 x double>* @vdr, align 16
  ret void
; CHECK-LABEL: @test1
; CHECK: xvdivdp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test2() {
entry:
  %0 = load <4 x float>, <4 x float>* @vfa, align 16
  %1 = load <4 x float>, <4 x float>* @vfb, align 16
  %2 = call <4 x float> @llvm.ppc.vsx.xvdivsp(<4 x float> %0, <4 x float> %1)
  store <4 x float> %2, <4 x float>* @vfr, align 16
  ret void
; CHECK-LABEL: @test2
; CHECK: xvdivsp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test3() {
entry:
  %0 = load <2 x double>, <2 x double>* @vda, align 16
  %1 = load <2 x double>, <2 x double>* @vda, align 16
  %2 = call <2 x double> @llvm.ceil.v2f64(<2 x double> %1)
  store <2 x double> %2, <2 x double>* @vdr, align 16
  ret void
; CHECK-LABEL: @test3
; CHECK: xvrdpip {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test4() {
entry:
  %0 = load <4 x float>, <4 x float>* @vfa, align 16
  %1 = load <4 x float>, <4 x float>* @vfa, align 16
  %2 = call <4 x float> @llvm.ceil.v4f32(<4 x float> %1)
  store <4 x float> %2, <4 x float>* @vfr, align 16
  ret void
; CHECK-LABEL: @test4
; CHECK: xvrspip {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test5() {
entry:
  %0 = load <2 x double>, <2 x double>* @vda, align 16
  %1 = load <2 x double>, <2 x double>* @vdb, align 16
  %2 = call <2 x i64> @llvm.ppc.vsx.xvcmpeqdp(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @vbllr, align 16
  ret void
; CHECK-LABEL: @test5
; CHECK: xvcmpeqdp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test6() {
entry:
  %0 = load <4 x float>, <4 x float>* @vfa, align 16
  %1 = load <4 x float>, <4 x float>* @vfb, align 16
  %2 = call <4 x i32> @llvm.ppc.vsx.xvcmpeqsp(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @vbir, align 16
  ret void
; CHECK-LABEL: @test6
; CHECK: xvcmpeqsp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test7() {
entry:
  %0 = load <2 x double>, <2 x double>* @vda, align 16
  %1 = load <2 x double>, <2 x double>* @vdb, align 16
  %2 = call <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @vbllr, align 16
  ret void
; CHECK-LABEL: @test7
; CHECK: xvcmpgedp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test8() {
entry:
  %0 = load <4 x float>, <4 x float>* @vfa, align 16
  %1 = load <4 x float>, <4 x float>* @vfb, align 16
  %2 = call <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @vbir, align 16
  ret void
; CHECK-LABEL: @test8
; CHECK: xvcmpgesp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test9() {
entry:
  %0 = load <2 x double>, <2 x double>* @vda, align 16
  %1 = load <2 x double>, <2 x double>* @vdb, align 16
  %2 = call <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double> %0, <2 x double> %1)
  store <2 x i64> %2, <2 x i64>* @vbllr, align 16
  ret void
; CHECK-LABEL: @test9
; CHECK: xvcmpgtdp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define void @test10() {
entry:
  %0 = load <4 x float>, <4 x float>* @vfa, align 16
  %1 = load <4 x float>, <4 x float>* @vfb, align 16
  %2 = call <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float> %0, <4 x float> %1)
  store <4 x i32> %2, <4 x i32>* @vbir, align 16
  ret void
; CHECK-LABEL: @test10
; CHECK: xvcmpgtsp {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define <4 x float> @emit_xvresp(<4 x float> %a) {
entry:
  %a.addr = alloca <4 x float>, align 16
  store <4 x float> %a, <4 x float>* %a.addr, align 16
  %0 = load <4 x float>, <4 x float>* %a.addr, align 16
  %1 = call <4 x float> @llvm.ppc.vsx.xvresp(<4 x float> %0)
  ret <4 x float> %1
; CHECK-LABEL: @emit_xvresp
; CHECK: xvresp {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind
define <2 x double> @emit_xvredp(<2 x double> %a) {
entry:
  %a.addr = alloca <2 x double>, align 16
  store <2 x double> %a, <2 x double>* %a.addr, align 16
  %0 = load <2 x double>, <2 x double>* %a.addr, align 16
  %1 = call <2 x double> @llvm.ppc.vsx.xvredp(<2 x double> %0)
  ret <2 x double> %1
; CHECK-LABEL: @emit_xvredp
; CHECK: xvredp {{[0-9]+}}, {{[0-9]+}}
}

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ppc.vsx.xvresp(<4 x float>)

; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ppc.vsx.xvredp(<2 x double>)

; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ceil.v2f64(<2 x double>)

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ceil.v4f32(<4 x float>)

; Function Attrs: nounwind readnone
declare <2 x double> @llvm.ppc.vsx.xvdivdp(<2 x double>, <2 x double>)

; Function Attrs: nounwind readnone
declare <4 x float> @llvm.ppc.vsx.xvdivsp(<4 x float>, <4 x float>)

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.vsx.xvcmpeqdp(<2 x double>, <2 x double>)

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.xvcmpeqsp(<4 x float>, <4 x float>)

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.vsx.xvcmpgedp(<2 x double>, <2 x double>)

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.xvcmpgesp(<4 x float>, <4 x float>)

; Function Attrs: nounwind readnone
declare <2 x i64> @llvm.ppc.vsx.xvcmpgtdp(<2 x double>, <2 x double>)

; Function Attrs: nounwind readnone
declare <4 x i32> @llvm.ppc.vsx.xvcmpgtsp(<4 x float>, <4 x float>)
