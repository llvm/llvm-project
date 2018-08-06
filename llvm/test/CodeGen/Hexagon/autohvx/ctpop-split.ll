; RUN: llc -march=hexagon < %s | FileCheck %s

; Check if popcounts of vector pairs are properly split.

; CHECK-LABEL: f0:
; CHECK: v0.h = vpopcount(v0.h)
; CHECK: v1.h = vpopcount(v1.h)
define <64 x i16> @f0(<64 x i16> %a0) #0 {
  %t0 = call <64 x i16> @llvm.ctpop.v64i32(<64 x i16> %a0)
  ret <64 x i16> %t0
}

; CHECK-LABEL: f1:
; CHECK: v0.h = vpopcount(v0.h)
; CHECK: v1.h = vpopcount(v1.h)
define <128 x i16> @f1(<128 x i16> %a0) #1 {
  %t0 = call <128 x i16> @llvm.ctpop.v128i32(<128 x i16> %a0)
  ret <128 x i16> %t0
}

declare <64 x i16>  @llvm.ctpop.v64i32(<64 x i16>) #0
declare <128 x i16>  @llvm.ctpop.v128i32(<128 x i16>) #1

attributes #0 = { readnone nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b,-packets" }
attributes #1 = { readnone nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b,-packets" }

