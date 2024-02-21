
; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#INT8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#INT64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#VINT8:]] = OpTypeVector %[[#INT8]] 2
; CHECK-DAG: %[[#PTRINT8:]] = OpTypePointer Workgroup %[[#INT8]]
; CHECK-DAG: %[[#PTRVINT8:]] = OpTypePointer Workgroup %[[#VINT8]]
; CHECK-DAG: %[[#CONST:]] = OpConstant %[[#INT64]] 1

; CHECK: %[[#PARAM1:]] = OpFunctionParameter %[[#PTRVINT8]]
define spir_kernel void @test1(ptr addrspace(3) %address) !kernel_arg_type !1 {
; CHECK: %[[#BITCAST1:]] = OpBitcast %[[#PTRINT8]] %[[#PARAM1]]
; CHECK: %[[#]] = OpInBoundsPtrAccessChain %[[#PTRINT8]] %[[#BITCAST1]] %[[#CONST]]
  %cast = bitcast ptr addrspace(3) %address to ptr addrspace(3)
  %gep = getelementptr inbounds i8, ptr addrspace(3) %cast, i64 1
  ret void
}

; CHECK: %[[#PARAM2:]] = OpFunctionParameter %[[#PTRVINT8]]
define spir_kernel void @test2(ptr addrspace(3) %address) !kernel_arg_type !1 {
; CHECK: %[[#BITCAST2:]] = OpBitcast %[[#PTRINT8]] %[[#PARAM2]]
; CHECK: %[[#]] = OpInBoundsPtrAccessChain %[[#PTRINT8]] %[[#BITCAST2]] %[[#CONST]]
  %gep = getelementptr inbounds i8, ptr addrspace(3) %address, i64 1
  ret void
}

declare spir_func <2 x i8> @_Z6vload2mPU3AS3Kc(i64, ptr addrspace(3))

!1 = !{!"char2*"}
