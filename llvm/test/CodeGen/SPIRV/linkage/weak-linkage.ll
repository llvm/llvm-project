; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_AMD_weak_linkage %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV-EXT
; RUNx: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_AMD_weak_linkage %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-EXT: Capability Linkage
; CHECK-SPIRV-EXT: Extension "SPV_AMD_weak_linkage"
; CHECK-SPIRV-EXT-DAG: OpDecorate %[[#]] LinkageAttributes "GV" Weak
; CHECK-SPIRV-EXT-DAG: OpDecorate %[[#]] LinkageAttributes "square" Weak

; CHECK-SPIRV-NOT: OpExtension "SPV_AMD_weak_linkage"
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "GV" Weak
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "square" Weak

@GV = weak addrspace(1) global [3 x i32] zeroinitializer, align 4

define spir_kernel void @k() {
entry:
  %call = call spir_func i32 @square(i32 2)
  ret void
}

define weak dso_local spir_func i32 @square(i32 %in) {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, ptr %in.addr, align 4
  %0 = load i32, ptr %in.addr, align 4
  %1 = load i32, ptr %in.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}
