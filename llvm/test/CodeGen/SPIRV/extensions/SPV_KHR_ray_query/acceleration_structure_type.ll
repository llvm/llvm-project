; Check that target("spirv.AccelerationStructureKHR") lowers to
; OpTypeAccelerationStructureKHR, gated by the RayQueryKHR capability and the
; SPV_KHR_ray_query extension.

; RUN: not llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute --spirv-ext=+SPV_KHR_ray_query %s -o - | FileCheck %s

; CHECK-ERROR: LLVM ERROR: OpTypeAccelerationStructureKHR type requires the following SPIR-V extension: SPV_KHR_ray_query

; CHECK-DAG: OpCapability RayQueryKHR
; CHECK-DAG: OpExtension "SPV_KHR_ray_query"
; CHECK-DAG: {{%[0-9]+}} = OpTypeAccelerationStructureKHR

define spir_func void @use_accel(target("spirv.AccelerationStructureKHR") %as) {
entry:
  ret void
}
