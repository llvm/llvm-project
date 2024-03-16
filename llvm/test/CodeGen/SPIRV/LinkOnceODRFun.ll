; RUN: llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=+SPV_KHR_linkonce_odr %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV-EXT
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-extensions=+SPV_KHR_linkonce_odr %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV-EXT: Capability Linkage
; CHECK-SPIRV-EXT: Extension "SPV_KHR_linkonce_odr"
; CHECK-SPIRV-EXT-DAG: OpDecorate %[[#]] LinkageAttributes "square" LinkOnceODR

define spir_kernel void @k() {
entry:
  %call = call spir_func i32 @square(i32 2)
  ret void
}

define linkonce_odr dso_local spir_func i32 @square(i32 %in) {
entry:
  ret i32 %in
}
