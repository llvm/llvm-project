; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=VK
; FIXME(135165) Alignment capability emitted for Vulkan.
; FIXME: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

; CL-DAG: %[[#Char:]] = OpTypeInt 8 0
; CL-DAG: %[[#PtrChar:]] = OpTypePointer Function %[[#Char]]

%tprange = type { %tparray }
%tparray = type { [2 x i64] }

; CL:      OpFunction
; CL:      %[[#FooVar:]] = OpVariable
; CL-NEXT: %[[#Casted1:]] = OpBitcast %[[#PtrChar]] %[[#FooVar]]
; CL-NEXT: OpLifetimeStart %[[#Casted1]], 72
; CL-NEXT: OpCopyMemorySized
; CL-NEXT: OpBitcast
; CL-NEXT: OpInBoundsPtrAccessChain
; CL-NEXT: %[[#Casted2:]] = OpBitcast %[[#PtrChar]] %[[#FooVar]]
; CL-NEXT: OpLifetimeStop %[[#Casted2]], 72

; VK:      OpFunction
; VK:      %[[#FooVar:]] = OpVariable
; VK-NEXT: OpCopyMemorySized
; VK-NEXT: OpInBoundsAccessChain
; VK-NEXT: OpReturn
define spir_func void @foo(ptr noundef byval(%tprange) align 8 %_arg_UserRange) {
  %RoundedRangeKernel = alloca %tprange, align 8
  call void @llvm.lifetime.start.p0(i64 72, ptr nonnull %RoundedRangeKernel)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %RoundedRangeKernel, ptr align 8 %_arg_UserRange, i64 16, i1 false)
  %KernelFunc = getelementptr inbounds i8, ptr %RoundedRangeKernel, i64 16
  call void @llvm.lifetime.end.p0(i64 72, ptr nonnull %RoundedRangeKernel)
  ret void
}

; CL: OpFunction
; CL: %[[#BarVar:]] = OpVariable
; CL-NEXT: OpLifetimeStart %[[#BarVar]], 0
; CL-NEXT: OpCopyMemorySized
; CL-NEXT: OpBitcast
; CL-NEXT: OpInBoundsPtrAccessChain
; CL-NEXT: OpLifetimeStop %[[#BarVar]], 0

; VK:      OpFunction
; VK:      %[[#BarVar:]] = OpVariable
; VK-NEXT: OpCopyMemorySized
; VK-NEXT: OpInBoundsAccessChain
; VK-NEXT: OpReturn
define spir_func void @bar(ptr noundef byval(%tprange) align 8 %_arg_UserRange) {
  %RoundedRangeKernel = alloca %tprange, align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %RoundedRangeKernel)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %RoundedRangeKernel, ptr align 8 %_arg_UserRange, i64 16, i1 false)
  %KernelFunc = getelementptr inbounds i8, ptr %RoundedRangeKernel, i64 16
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %RoundedRangeKernel)
  ret void
}

; CL: OpFunction
; CL: %[[#TestVar:]] = OpVariable
; CL-NEXT: OpLifetimeStart %[[#TestVar]], 1
; CL-NEXT: OpCopyMemorySized
; CL-NEXT: OpInBoundsPtrAccessChain
; CL-NEXT: OpLifetimeStop %[[#TestVar]], 1

; VK:      OpFunction
; VK:      %[[#Test:]] = OpVariable
; VK-NEXT: OpCopyMemorySized
; VK-NEXT: OpInBoundsAccessChain
; VK-NEXT: OpReturn
define spir_func void @test(ptr noundef align 8 %_arg) {
  %var = alloca i8, align 8
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %var)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %var, ptr align 8 %_arg, i64 1, i1 false)
  %KernelFunc = getelementptr inbounds i8, ptr %var, i64 0
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %var)
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
