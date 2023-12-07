; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Check that the vsplat instruction is generated
; CHECK: .word 1097875824
; CHECK: .word 1048133241
; CHECK: .word 0

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"
; Function Attrs: nofree norecurse nounwind writeonly
define dso_local i32 @foo(ptr nocapture %a) local_unnamed_addr #0 {
vector.body:
  store <40 x half> <half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH4170, half 0xH3E79, half 0xH3E79, half 0xH3E79, half 0xH3E79, half 0xH3E79, half 0xH3E79, half 0xH3E79, half 0xH3E79, half 0xH3E79>, ptr %a, align 2
  ret i32 0
}

attributes #0 = { nofree norecurse nounwind writeonly "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv69" "target-features"="+hvx-length128b,+hvxv69,+v69,-long-calls" "unsafe-fp-math"="false" "use-soft-float"="false" }
