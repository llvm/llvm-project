; RUN: opt -S -dxil-metadata-emit < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-compute"

;CHECK:!dx.entryPoints = !{![[entry:[0-9]+]]}

;CHECK:![[entry]] = !{ptr @entry, !"entry", null, null, ![[extra:[0-9]+]]}
;CHECK:![[extra]] = !{i32 4, ![[numthreads:[0-9]+]]}
;CHECK:![[numthreads]] = !{i32 1, i32 2, i32 1}

; Function Attrs: noinline nounwind
define void @entry() #0 {
entry:
  ret void
}

attributes #0 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
