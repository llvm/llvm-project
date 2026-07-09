; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_AMD_weak_linkage %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV-EXT
; TODO: enable validation when SPIR-V Headers patch is merged
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_AMD_weak_linkage %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-EXT-DAG: Capability Linkage
; CHECK-SPIRV-EXT-DAG: Capability WeakLinkageAMD
; CHECK-SPIRV-EXT: Extension "SPV_AMD_weak_linkage"
; CHECK-SPIRV-EXT-DAG: OpDecorate %[[#]] LinkageAttributes "GV" WeakAMD
; CHECK-SPIRV-EXT-DAG: OpDecorate %[[#]] LinkageAttributes "square" WeakAMD

; CHECK-SPIRV-NOT: OpExtension "SPV_AMD_weak_linkage"
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "GV" WeakAMD
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "square" WeakAMD
; CHECK-SPIRV-DAG: OpDecorate %[[#]] LinkageAttributes "GV" Export
; CHECK-SPIRV-DAG: OpDecorate %[[#]] LinkageAttributes "square" Export

@GV = weak addrspace(1) global [3 x i32] zeroinitializer, align 4

define spir_kernel void @k() {
entry:
  %call = call spir_func i32 @square(i32 2)
  ret void
}

define weak dso_local spir_func i32 @square(i32 %in) {
entry:
  %mul = mul nsw i32 %in, %in
  ret i32 %mul
}
