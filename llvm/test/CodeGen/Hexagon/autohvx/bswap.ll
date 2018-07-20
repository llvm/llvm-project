; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00
; CHECK: [[R00:r[0-9]+]] = ##16843009
; CHECK: [[V00:v[0-9]+]] = vsplat([[R00]])
; CHECK: v0 = vdelta(v0,[[V00]])
define <32 x i16> @test_00(<32 x i16> %a0) #0 {
  %v0 = call <32 x i16> @llvm.bswap.v32i16(<32 x i16> %a0)
  ret <32 x i16> %v0
}

; CHECK-LABEL: test_01
; CHECK: [[R01:r[0-9]+]] = ##50529027
; CHECK: [[V01:v[0-9]+]] = vsplat([[R01]])
; CHECK: v0 = vdelta(v0,[[V01]])
define <16 x i32> @test_01(<16 x i32> %a0) #0 {
  %v0 = call <16 x i32> @llvm.bswap.v16i32(<16 x i32> %a0)
  ret <16 x i32> %v0
}

; CHECK-LABEL: test_10
; CHECK: [[R10:r[0-9]+]] = ##16843009
; CHECK: [[V10:v[0-9]+]] = vsplat([[R10]])
; CHECK: v0 = vdelta(v0,[[V10]])
define <64 x i16> @test_10(<64 x i16> %a0) #1 {
  %v0 = call <64 x i16> @llvm.bswap.v64i16(<64 x i16> %a0)
  ret <64 x i16> %v0
}

; CHECK-LABEL: test_11
; CHECK: [[R11:r[0-9]+]] = ##50529027
; CHECK: [[V11:v[0-9]+]] = vsplat([[R11]])
; CHECK: v0 = vdelta(v0,[[V11]])
define <32 x i32> @test_11(<32 x i32> %a0) #1 {
  %v0 = call <32 x i32> @llvm.bswap.v32i32(<32 x i32> %a0)
  ret <32 x i32> %v0
}

declare <32 x i16> @llvm.bswap.v32i16(<32 x i16>) #0
declare <16 x i32> @llvm.bswap.v16i32(<16 x i32>) #0
declare <64 x i16> @llvm.bswap.v64i16(<64 x i16>) #1
declare <32 x i32> @llvm.bswap.v32i32(<32 x i32>) #1

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b" }
