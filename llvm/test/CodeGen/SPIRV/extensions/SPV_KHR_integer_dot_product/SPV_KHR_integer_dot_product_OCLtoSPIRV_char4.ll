; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32v1.6-unknown-unknown %s -o - -filetype=obj | spirv-val %}
; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s --check-prefixes=CHECK,CHECK-EXT
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability DotProduct
; CHECK: Capability DotProductInput4x8Bit
; CHECK-EXT: OpExtension "SPV_KHR_integer_dot_product"
; CHECK-NOT: OpExtension "SPV_KHR_integer_dot_product"

; CHECK: Name %[[#SignedA:]] "ia"
; CHECK: Name %[[#UnsignedA:]] "ua"
; CHECK: Name %[[#SignedB:]] "ib"
; CHECK: Name %[[#UnsignedB:]] "ub"

; CHECK: SDot %[[#]] %[[#SignedA]] %[[#SignedB]]
; CHECK: SUDot %[[#]] %[[#SignedA]] %[[#UnsignedB]]
; CHECK: SUDot %[[#]] %[[#SignedB]] %[[#UnsignedA]]
; CHECK: UDot %[[#]] %[[#UnsignedA]] %[[#UnsignedB]]

; CHECK: SDotAccSat %[[#]] %[[#SignedA]] %[[#SignedB]] %[[#]]
; CHECK: SUDotAccSat %[[#]] %[[#SignedA]] %[[#UnsignedB]] %[[#]]
; CHECK: SUDotAccSat %[[#]] %[[#SignedB]] %[[#UnsignedA]] %[[#]]
; CHECK: UDotAccSat %[[#]] %[[#UnsignedA]] %[[#UnsignedB]] %[[#]]

define spir_kernel void @test(<4 x i8> %ia, <4 x i8> %ua, <4 x i8> %ib, <4 x i8> %ub, <4 x i8> %ires, <4 x i8> %ures) {
entry:
  %call = tail call spir_func i32 @_Z3dotDv4_cS_(<4 x i8> %ia, <4 x i8> %ib) #2
  %call1 = tail call spir_func i32 @_Z3dotDv4_cDv4_h(<4 x i8> %ia, <4 x i8> %ub) #2
  %call2 = tail call spir_func i32 @_Z3dotDv4_hDv4_c(<4 x i8> %ua, <4 x i8> %ib) #2
  %call3 = tail call spir_func i32 @_Z3dotDv4_hS_(<4 x i8> %ua, <4 x i8> %ub) #2
  %call4 = tail call spir_func i32 @_Z11dot_acc_satDv4_cS_i(<4 x i8> %ia, <4 x i8> %ib, i32 %call2) #2
  %call5 = tail call spir_func i32 @_Z11dot_acc_satDv4_cDv4_hi(<4 x i8> %ia, <4 x i8> %ub, i32 %call4) #2
  %call6 = tail call spir_func i32 @_Z11dot_acc_satDv4_hDv4_ci(<4 x i8> %ua, <4 x i8> %ib, i32 %call5) #2
  %call7 = tail call spir_func i32 @_Z11dot_acc_satDv4_hS_j(<4 x i8> %ua, <4 x i8> %ub, i32 %call3) #2
  ret void
}

declare spir_func i32 @_Z3dotDv4_cS_(<4 x i8>, <4 x i8>)
declare spir_func i32 @_Z3dotDv4_cDv4_h(<4 x i8>, <4 x i8>)
declare spir_func i32 @_Z3dotDv4_hDv4_c(<4 x i8>, <4 x i8>)
declare spir_func i32 @_Z3dotDv4_hS_(<4 x i8>, <4 x i8>)
declare spir_func i32 @_Z11dot_acc_satDv4_cS_i(<4 x i8>, <4 x i8>, i32)
declare spir_func i32 @_Z11dot_acc_satDv4_cDv4_hi(<4 x i8>, <4 x i8>, i32)
declare spir_func i32 @_Z11dot_acc_satDv4_hDv4_ci(<4 x i8>, <4 x i8>, i32)
declare spir_func i32 @_Z11dot_acc_satDv4_hS_j(<4 x i8>, <4 x i8>, i32)

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
