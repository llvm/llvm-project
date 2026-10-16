; RUN: not llc -O0 -mtriple=spirv1.5-unknown-vulkan1.2-compute %s -o /dev/null 2>&1 | FileCheck %s
; CHECK: error: {{.*}}llvm.spv.debug.break requires SPIR-V 1.6 or the following SPIR-V extension: SPV_KHR_non_semantic_info

define void @main() #0 {
entry:
  call void @llvm.spv.debug.break()
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
