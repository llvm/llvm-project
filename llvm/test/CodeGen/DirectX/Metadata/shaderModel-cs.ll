; RUN: opt -S -dxil-translate-metadata %s | FileCheck %s
; RUN: opt -S -dxil-prepare  %s | FileCheck %s  --check-prefix=REMOVE_EXTRA_ATTRIBUTE

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"cs", i32 6, i32 6}

define void @entry() #0 {
entry:
  ret void
}

; Make sure extra attribute like hlsl.numthreads are removed.
; And experimental attribute is removed when validator version is not 0.0.
; REMOVE_EXTRA_ATTRIBUTE:attributes #0 = { noinline nounwind }
attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
