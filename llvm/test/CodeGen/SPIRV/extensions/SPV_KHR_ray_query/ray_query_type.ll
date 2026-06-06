; Check that target("spirv.RayQueryKHR") lowers to OpTypeRayQueryKHR under the
; Vulkan flavor, gated behind the RayQueryKHR capability and the SPV_KHR_ray_query
; extension, and that it errors cleanly without the extension enabled.

; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_ray_query %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: OpTypeRayQueryKHR type requires the following SPIR-V extension: SPV_KHR_ray_query

; CHECK-DAG: OpCapability RayQueryKHR
; CHECK-DAG: OpExtension "SPV_KHR_ray_query"
; CHECK-DAG: {{%[0-9]+}} = OpTypeRayQueryKHR

; A by-value parameter of the opaque type forces OpTypeRayQueryKHR into the module
; (referenced by OpTypeFunction — cannot be eliminated like an unused local).
define spir_func void @use_ray_query(target("spirv.RayQueryKHR") %rq) {
entry:
  ret void
}
