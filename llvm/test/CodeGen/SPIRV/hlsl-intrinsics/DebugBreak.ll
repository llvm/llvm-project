; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=CHECK,CORE
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.6-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}
; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv1.5-unknown-vulkan1.2-compute --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s --check-prefixes=CHECK,EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv1.5-unknown-vulkan1.2-compute --spirv-ext=+SPV_KHR_non_semantic_info %s -o - -filetype=obj | spirv-val --target-env vulkan1.2 %}

; CORE-NOT: OpExtension "SPV_KHR_non_semantic_info"
; EXT: OpExtension "SPV_KHR_non_semantic_info"
; CHECK: %[[#set:]] = OpExtInstImport "NonSemantic.DebugBreak"
; CHECK-DAG: %[[#void:]] = OpTypeVoid
; CHECK: OpFunction
; CHECK: %[[#]] = OpExtInst %[[#void]] %[[#set]] DebugBreak
; CHECK-NEXT: OpReturn

define void @main() #0 {
entry:
  call void @llvm.spv.debug.break()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
