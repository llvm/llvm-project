; RUN: llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-SPIRV: %[[#int:]] = OpTypeInt 32
; CHECK-SPIRV: OpBitReverse %[[#int]]

define spir_func void @testBitRev(i32 %a, i32 %b, i32 %c, ptr %res) local_unnamed_addr {
entry:
  %call = tail call i32 @llvm.bitreverse.i32(i32 %b)
  store i32 %call, ptr %res, align 4
  ret void
}

declare i32 @llvm.bitreverse.i32(i32)
