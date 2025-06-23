; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that we generate a proper vinsert instruction for f16 types.
; CHECK: vinsert
target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define ptr @fred(ptr %v0) local_unnamed_addr #0 {
b0:
  %v1 = load <64 x half>, ptr %v0, align 2
  %v2 = insertelement <64 x half> %v1, half 0xH4170, i32 17
  store volatile <64 x half> %v2, ptr %v0, align 2
  ret ptr %v0
}

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv69" "target-features"="+hvx-length128b,+hvxv69,+v69,+hvx-qfloat,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
