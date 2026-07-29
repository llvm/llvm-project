; HLSL `groupshared` variables are emitted by clang as external, hidden globals
; in the Workgroup storage class (a declaration, since shared memory has no
; initializer). getSpirvLinkageTypeFor must NOT give such module-internal
; storage-class declarations Import linkage: doing so decorates them with
; LinkageAttributes and forces OpCapability Linkage, which is illegal in a
; Vulkan shader. Verify no Linkage capability / decoration is emitted while the
; variable is still materialized as a Workgroup OpVariable.

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=CHECK,NOLINK
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; The groupshared variable is still emitted, in the Workgroup storage class.
; CHECK-DAG: OpName %[[#gs:]] "gs"
; CHECK-DAG: %[[#PtrWG:]] = OpTypePointer Workgroup %{{[0-9]+}}
; CHECK-DAG: %[[#gs]] = OpVariable %[[#PtrWG]] Workgroup

; No Linkage capability and no LinkageAttributes decoration may be present.
; A NOT-only prefix scans the whole module (the capability and decoration
; sections precede the OpVariable definition in SPIR-V layout).
; NOLINK-NOT: OpCapability Linkage
; NOLINK-NOT: LinkageAttributes

@gs = external hidden addrspace(3) global i32, align 4

define void @main() #0 {
entry:
  store i32 0, ptr addrspace(3) @gs, align 4
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
