; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - | FileCheck %s --check-prefixes=VK
; FIXME(182779) ByVal attribute emitted for Vulkan.
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv-unknown-vulkan1.3-compute %s -o - -filetype=obj | spirv-val %}

%tprange = type { %tparray }
%tparray = type { [2 x i64] }

; CL:      OpFunction
; CL:      %[[#FooVar:]] = OpVariable
; CL-NEXT: OpLifetimeStart %[[#FooVar]] 0
; CL: OpInBoundsPtrAccessChain
; CL: OpLifetimeStop %[[#FooVar]] 0

; VK:      OpFunction
; VK:      %[[#FooVar:]] = OpVariable
; VK-NEXT: OpInBoundsAccessChain
; VK-NEXT: OpStore
; VK-NEXT: OpReturn
define spir_func void @foo(ptr noundef byval(%tprange) align 8 %_arg_UserRange) {
  %RoundedRangeKernel = alloca %tprange, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %RoundedRangeKernel)
  %KernelFunc = getelementptr inbounds i8, ptr %RoundedRangeKernel, i64 8
  store i64 zeroinitializer, ptr %KernelFunc, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %RoundedRangeKernel)
  ret void
}

; CL: OpFunction
; CL: %[[#BarVar:]] = OpVariable
; CL-NEXT: OpLifetimeStart %[[#BarVar]] 0
; CL: OpInBoundsPtrAccessChain
; CL: OpLifetimeStop %[[#BarVar]] 0

; VK:      OpFunction
; VK:      %[[#BarVar:]] = OpVariable
; VK-NEXT: OpInBoundsAccessChain
; VK-NEXT: OpStore
; VK-NEXT: OpReturn
define spir_func void @bar(ptr noundef byval(%tprange) align 8 %_arg_UserRange) {
  %RoundedRangeKernel = alloca %tprange, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %RoundedRangeKernel)
  %KernelFunc = getelementptr inbounds i8, ptr %RoundedRangeKernel, i64 8
  store i64 zeroinitializer, ptr %KernelFunc, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %RoundedRangeKernel)
  ret void
}

; CL: OpFunction
; CL: %[[#TestVar:]] = OpVariable
; CL: OpLifetimeStart %[[#TestVar]] 1
; CL: OpInBoundsPtrAccessChain
; CL: OpLifetimeStop %[[#TestVar]] 1

; VK:      OpFunction
; VK:      %[[#Test:]] = OpVariable
; VK-NEXT: OpInBoundsAccessChain
; VK-NEXT: OpStore
; VK-NEXT: OpReturn
define spir_func void @test(ptr noundef align 8 %_arg) {
  %var = alloca i8, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %var)
  %KernelFunc = getelementptr inbounds i8, ptr %var, i64 1
  store i8 0, ptr %KernelFunc, align 8
  call void @llvm.lifetime.end.p0(ptr nonnull %var)
  ret void
}

; An array of i8 is not an i8 pointee, so Size must be 0.
; CL: OpFunction
; CL: %[[#ByteArrVar:]] = OpVariable
; CL-NEXT: OpLifetimeStart %[[#ByteArrVar]] 0
; CL: OpLifetimeStop %[[#ByteArrVar]] 0
define spir_func void @byte_array() {
  %var = alloca [16 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %var)
  store [16 x i8] zeroinitializer, ptr %var, align 1
  call void @llvm.lifetime.end.p0(ptr nonnull %var)
  ret void
}

declare void @llvm.lifetime.start.p0(ptr nocapture)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.lifetime.end.p0(ptr nocapture)
