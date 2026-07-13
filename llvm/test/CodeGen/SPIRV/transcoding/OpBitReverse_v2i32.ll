; RUN: llc -O0 -mtriple=spirv-unknown-vulkan %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-SPIRV: %[[#short:]] = OpTypeInt 32
; CHECK-SPIRV: %[[#short2:]] = OpTypeVector %[[#short]] 2
; CHECK-SPIRV: OpBitReverse %[[#short2]]

define hidden spir_func void @testBitRev(<2 x i32> %a, <2 x i32> %b, <2 x i32> %c, ptr %res) {
entry:
  %call = tail call <2 x i32> @llvm.bitreverse.v2i32(<2 x i32> %b)
  store <2 x i32> %call, ptr %res, align 4
  ret void
}

define void @main() #1 {
  ret void
}

declare <2 x i32> @llvm.bitreverse.v2i32(<2 x i32>)
attributes #1 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
