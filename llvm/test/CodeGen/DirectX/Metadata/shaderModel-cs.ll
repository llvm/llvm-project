; RUN: opt -S -dxil-metadata-emit %s | FileCheck %s
; RUN: opt -S -dxil-prepare  %s | FileCheck %s  --check-prefix=REMOVE_EXTRA_ATTRIBUTE
; RUN: opt -S -passes="print<dxil-metadata>" -disable-output %s 2>&1 | FileCheck %s --check-prefix=ANALYSIS

target triple = "dxil-pc-shadermodel6.6-compute"

; CHECK: !dx.shaderModel = !{![[SM:[0-9]+]]}
; CHECK: ![[SM]] = !{!"cs", i32 6, i32 6}

; ANALYSIS: Shader Model Version : 6.6
; ANALYSIS: DXIL Version : 1.6
; ANALYSIS: Shader Stage : compute

define void @entry() #0 {
entry:
  ret void
}

; Make sure extra attribute like hlsl.numthreads are removed.
; And experimental attribute is removed when validator version is not 0.0.
; REMOVE_EXTRA_ATTRIBUTE:attributes #0 = { noinline nounwind }
attributes #0 = { noinline nounwind "exp-shader"="cs" "hlsl.numthreads"="1,2,1" "hlsl.shader"="compute" }
