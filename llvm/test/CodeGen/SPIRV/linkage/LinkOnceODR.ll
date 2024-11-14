; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_linkonce_odr %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV-EXT
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_linkonce_odr %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-EXT: Capability Linkage
; CHECK-SPIRV-EXT: Extension "SPV_KHR_linkonce_odr"
; CHECK-SPIRV-EXT-DAG: OpDecorate %[[#]] LinkageAttributes "GV" LinkOnceODR
; CHECK-SPIRV-EXT-DAG: OpDecorate %[[#]] LinkageAttributes "square" LinkOnceODR

; No extension -> no LinkOnceODR
; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-NOT: OpExtension "SPV_KHR_linkonce_odr"
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "GV" LinkOnceODR 
; CHECK-SPIRV-NOT: OpDecorate %[[#]] LinkageAttributes "square" LinkOnceODR 

@GV = linkonce_odr addrspace(1) global [3 x i32] zeroinitializer, align 4

define spir_kernel void @k() {
entry:
  %call = call spir_func i32 @square(i32 2)
  ret void
}

define linkonce_odr dso_local spir_func i32 @square(i32 %in) {
entry:
  %in.addr = alloca i32, align 4
  store i32 %in, i32* %in.addr, align 4
  %0 = load i32, i32* %in.addr, align 4
  %1 = load i32, i32* %in.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}
