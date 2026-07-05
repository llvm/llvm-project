; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-SPIRV: %[[#int:]] = OpTypeInt 32
; CHECK-SPIRV: OpBitReverse %[[#int]]

define hidden spir_func void @testBitRev(i32 %a, i32 %b, i32 %c, ptr %res) {
entry:
  %call = tail call i32 @llvm.bitreverse.i32(i32 %b)
  store i32 %call, ptr %res, align 4
  ret void
}

define void @main() #1 {
  ret void
}

declare i32 @llvm.bitreverse.i32(i32)
attributes #1 = { "hlsl.numthreads"="8,1,1" "hlsl.shader"="compute" }
