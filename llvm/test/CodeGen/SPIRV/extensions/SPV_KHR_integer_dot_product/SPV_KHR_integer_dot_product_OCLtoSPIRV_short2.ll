; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown --spirv-ext=+SPV_KHR_integer_dot_product %s -o - -filetype=obj | spirv-val %}

; CHECK: Capability DotProduct
; CHECK: Capability DotProductInputAll

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

define spir_kernel void @test1(<2 x i16> %ia, <2 x i16> %ua, <2 x i16> %ib, <2 x i16> %ub, <2 x i16> %ires, <2 x i16> %ures) !kernel_arg_addr_space !3 !kernel_arg_access_qual !4 !kernel_arg_type !5 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %call = tail call spir_func i32 @_Z3dotDv2_sS_(<2 x i16> %ia, <2 x i16> %ib) #2
  %call1 = tail call spir_func i32 @_Z3dotDv2_sDv2_t(<2 x i16> %ia, <2 x i16> %ub) #2
  %call2 = tail call spir_func i32 @_Z3dotDv2_tDv2_s(<2 x i16> %ua, <2 x i16> %ib) #2
  %call3 = tail call spir_func i32 @_Z3dotDv2_tS_(<2 x i16> %ua, <2 x i16> %ub) #2
  %call4 = tail call spir_func i32 @_Z11dot_acc_satDv2_sS_i(<2 x i16> %ia, <2 x i16> %ib, i32 %call2) #2
  %call5 = tail call spir_func i32 @_Z11dot_acc_satDv2_sDv2_ti(<2 x i16> %ia, <2 x i16> %ub, i32 %call4) #2
  %call6 = tail call spir_func i32 @_Z11dot_acc_satDv2_tDv2_si(<2 x i16> %ua, <2 x i16> %ib, i32 %call5) #2
  %call7 = tail call spir_func i32 @_Z11dot_acc_satDv2_tS_j(<2 x i16> %ua, <2 x i16> %ub, i32 %call3) #2
  ret void
}

declare spir_func i32 @_Z3dotDv2_sS_(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z3dotDv2_sDv2_t(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z3dotDv2_tDv2_s(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z3dotDv2_tS_(<2 x i16>, <2 x i16>)
declare spir_func i32 @_Z11dot_acc_satDv2_sS_i(<2 x i16>, <2 x i16>, i32)
declare spir_func i32 @_Z11dot_acc_satDv2_sDv2_ti(<2 x i16>, <2 x i16>, i32)
declare spir_func i32 @_Z11dot_acc_satDv2_tDv2_si(<2 x i16>, <2 x i16>, i32)
declare spir_func i32 @_Z11dot_acc_satDv2_tS_j(<2 x i16>, <2 x i16>, i32)

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!3 = !{i32 0, i32 0, i32 0, i32 0, i32 0, i32 0}
!4 = !{!"none", !"none", !"none", !"none", !"none", !"none"}
!5 = !{!"short4", !"ushort4", !"short4", !"ushort4", !"short4", !"ushort4"}
!6 = !{!"short __attribute__((ext_vector_type(4)))", !"ushort __attribute__((ext_vector_type(4)))", !"short __attribute__((ext_vector_type(4)))", !"ushort __attribute__((ext_vector_type(4)))", !"short __attribute__((ext_vector_type(4)))", !"ushort __attribute__((ext_vector_type(4)))"}
!7 = !{!"", !"", !"", !"", !"", !""}
