; HLSL `groupshared` variables are emitted as external Workgroup declarations.
; getSpirvLinkageTypeFor must not give them Import linkage, which would add a
; LinkageAttributes decoration and force OpCapability Linkage (illegal in Vulkan).

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; The groupshared variable is still emitted, in the Workgroup storage class.
; CHECK-DAG: OpName %[[#gs:]] "gs"
; CHECK-DAG: %[[#PtrWG:]] = OpTypePointer Workgroup %{{[0-9]+}}
; CHECK-DAG: %[[#gs]] = OpVariable %[[#PtrWG]] Workgroup

; No Linkage capability and no LinkageAttributes decoration may be present.
; CHECK-NOT: OpCapability Linkage
; CHECK-NOT: LinkageAttributes

@gs = external hidden addrspace(3) global i32, align 4

define void @main() #0 {
entry:
  store i32 0, ptr addrspace(3) @gs, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
