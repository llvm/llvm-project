; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check that this testcase compiles successfully.
; CHECK: vextract

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon-unknown--elf"

define void @fred() local_unnamed_addr #0 {
b0:
  %v1 = load <64 x i8>, <64 x i8>* undef, align 64
  %v2 = insertelement <64 x i8> %v1, i8 0, i32 0
  br label %b3

b3:                                               ; preds = %b3, %b0
  %v4 = phi <64 x i8> [ %v2, %b0 ], [ %v6, %b3 ]
  %v5 = extractelement <64 x i8> %v4, i32 0
  %v6 = insertelement <64 x i8> %v4, i8 undef, i32 0
  br label %b3
}

attributes #0 = { "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
