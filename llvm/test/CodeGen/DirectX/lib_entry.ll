; RUN: opt -S -dxil-translate-metadata < %s | FileCheck %s
target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-unknown-shadermodel6.7-library"

;CHECK:!dx.entryPoints = !{![[empty_entry:[0-9]+]], ![[entry:[0-9]+]]}

; Make sure generate empty entry for lib profile.
;CHECK:![[empty_entry]] = !{null, !"", null, null, ![[shader_flags:[0-9]+]]}
; Make sure double is marked for shader flags.
;CHECK:![[shader_flags]] = !{i32 0, i64 4}
;CHECK:![[entry]] = !{ptr @entry, !"entry", null, null, ![[extra:[0-9]+]]}
;CHECK:![[extra]] = !{i32 8, i32 5, i32 4, ![[numthreads:[0-9]+]]}
;CHECK:![[numthreads]] = !{i32 1, i32 2, i32 1}

; Function Attrs: noinline nounwind
define void @entry() #0 {
entry:
  %0 = fpext float 2.000000e+00 to double
  ret void
}

attributes #0 = { noinline nounwind "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
