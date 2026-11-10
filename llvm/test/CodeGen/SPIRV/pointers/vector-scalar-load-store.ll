; RUN: llc -O0 -verify-machineinstrs -mtriple=spirv-unknown-vulkan-compute %s -o - | FileCheck %s
; RUN: opt -S -passes='spirv-emit-intrinsics,function(spirv-legalize-pointer-cast)' -mtriple=spirv-unknown-vulkan-compute %s | FileCheck %s --check-prefix=IR
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan-compute %s -o - -filetype=obj | spirv-val --target-env vulkan1.3 %}

; CHECK-DAG: %[[#I16:]] = OpTypeInt 16 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#V4I16:]] = OpTypeVector %[[#I16]] 4
; CHECK-DAG: %[[#PTR4:]] = OpTypePointer StorageBuffer %[[#V4I16]]

@.str = private unnamed_addr constant [4 x i8] c"Buf\00", align 1

define void @main() #0 {
; CHECK-LABEL: ; -- Begin function main
; CHECK: %[[#PTR:]] = OpAccessChain %[[#PTR4]]
; CHECK: %[[#LOAD:]] = OpLoad %[[#V4I16]] %[[#PTR]]
; CHECK: %[[#SCALAR:]] = OpBitcast %[[#I64]] %[[#LOAD]]
; CHECK: %[[#STORE:]] = OpBitcast %[[#V4I16]] %[[#SCALAR]]
; CHECK: OpStore %[[#PTR]] %[[#STORE]]
; IR-LABEL: define void @main(
; IR: load <4 x i16>, ptr addrspace(11) %{{.*}}, align 2
; IR: call i64 @llvm.spv.bitcast.i64.v4i16(
; IR: call <4 x i16> @llvm.spv.bitcast.v4i16.i64(
; IR: store <4 x i16> %{{.*}}, ptr addrspace(11) %{{.*}}, align 2
entry:
  %buffer = call target("spirv.VulkanBuffer", [0 x <4 x i16>], 12, 1) @llvm.spv.resource.handlefrombinding(i32 0, i32 0, i32 1, i32 0, ptr @.str)
  %ptr = call ptr addrspace(11) @llvm.spv.resource.getpointer(target("spirv.VulkanBuffer", [0 x <4 x i16>], 12, 1) %buffer, i32 0)
  %value = load i64, ptr addrspace(11) %ptr, align 2
  store i64 %value, ptr addrspace(11) %ptr, align 2
  ret void
}

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
