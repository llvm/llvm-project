; RUN: llc -mtriple=hexagon -mcpu=hexagonv73 -mattr=+hvxv73,+hvx-length128b \
; RUN:   < %s | FileCheck %s
;
; Check that extracting a small predicate subvector (<8 x i1) from an HVX
; predicate compiles correctly. The bug was in extractHvxSubvectorPred where
; the loop generating the shuffle mask used HwLen/ResLen instead of HwLen/8,
; producing a mask of wrong size for ResLen < 8.

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown-linux-musl"

; CHECK-LABEL: test_extract_v4i1:
; CHECK-DAG:   vand(v{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG:   vdelta(v{{[0-9]+}},v{{[0-9]+}})
; CHECK:       dealloc_return
define <4 x i1> @test_extract_v4i1(<128 x i1> %pred) {
  %r = shufflevector <128 x i1> %pred, <128 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i1> %r
}

; CHECK-LABEL: test_extract_v2i1:
; CHECK-DAG:   vand(v{{[0-9]+}},r{{[0-9]+}})
; CHECK-DAG:   vdelta(v{{[0-9]+}},v{{[0-9]+}})
; CHECK:       dealloc_return
define <2 x i1> @test_extract_v2i1(<128 x i1> %pred) {
  %r = shufflevector <128 x i1> %pred, <128 x i1> poison, <2 x i32> <i32 0, i32 1>
  ret <2 x i1> %r
}
