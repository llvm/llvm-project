; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PTRINT8:]] = OpTypePointer CrossWorkgroup %[[#INT8]]

define spir_kernel void @test_fn(ptr addrspace(1) %src) !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_type_qual !4 !kernel_arg_base_type !3 {
entry:
  %g1 = call spir_func i64 @_Z13get_global_idj(i32 0)
  %i1 = insertelement <3 x i64> undef, i64 %g1, i32 0
  %g2 = call spir_func i64 @_Z13get_global_idj(i32 1)
  %i2 = insertelement <3 x i64> %i1, i64 %g2, i32 1
  %g3 = call spir_func i64 @_Z13get_global_idj(i32 2)
  %i3 = insertelement <3 x i64> %i2, i64 %g3, i32 2
  %e = extractelement <3 x i64> %i3, i32 0
  %c1 = trunc i64 %e to i32
  %c2 = sext i32 %c1 to i64
  %b = bitcast ptr addrspace(1) %src to ptr addrspace(1)

; Make sure that builtin call directly uses either a OpBitcast or OpFunctionParameter of i8* type
; CHECK: %[[#BITCASTorPARAMETER:]] = {{OpBitcast|OpFunctionParameter}}{{.*}}%[[#PTRINT8]]{{.*}}
; CHECK: %[[#]] = OpExtInst %[[#]] %[[#]] vloadn %[[#]] %[[#BITCASTorPARAMETER]] 3
  %call = call spir_func <3 x i8> @_Z6vload3mPU3AS1Kc(i64 %c2, ptr addrspace(1) %b)

  ret void
}

declare spir_func i64 @_Z13get_global_idj(i32)

declare spir_func <3 x i8> @_Z6vload3mPU3AS1Kc(i64, ptr addrspace(1))

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"char3*"}
!4 = !{!""}
