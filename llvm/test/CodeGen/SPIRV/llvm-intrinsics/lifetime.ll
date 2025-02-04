; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrChar:]] = OpTypePointer Function %[[#Char]]

%tprange = type { %tparray }
%tparray = type { [2 x i64] }

; CHECK: OpFunction
; CHECK: %[[#FooVar:]] = OpVariable
; CHECK: %[[#Casted1:]] = OpBitcast %[[#PtrChar]] %[[#FooVar]]
; CHECK: OpLifetimeStart %[[#Casted1]], 72
; CHECK: OpCopyMemorySized
; CHECK: OpBitcast
; CHECK: OpInBoundsPtrAccessChain
; CHECK: %[[#Casted2:]] = OpBitcast %[[#PtrChar]] %[[#FooVar]]
; CHECK: OpLifetimeStop %[[#Casted2]], 72
define spir_func void @foo(ptr noundef byval(%tprange) align 8 %_arg_UserRange) {
  %RoundedRangeKernel = alloca %tprange, align 8
  call void @llvm.lifetime.start.p0(i64 72, ptr nonnull %RoundedRangeKernel)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %RoundedRangeKernel, ptr align 8 %_arg_UserRange, i64 16, i1 false)
  %KernelFunc = getelementptr inbounds i8, ptr %RoundedRangeKernel, i64 16
  call void @llvm.lifetime.end.p0(i64 72, ptr nonnull %RoundedRangeKernel)
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#BarVar:]] = OpVariable
; CHECK: OpLifetimeStart %[[#BarVar]], 0
; CHECK: OpCopyMemorySized
; CHECK: OpBitcast
; CHECK: OpInBoundsPtrAccessChain
; CHECK: OpLifetimeStop %[[#BarVar]], 0
define spir_func void @bar(ptr noundef byval(%tprange) align 8 %_arg_UserRange) {
  %RoundedRangeKernel = alloca %tprange, align 8
  call void @llvm.lifetime.start.p0(i64 -1, ptr nonnull %RoundedRangeKernel)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %RoundedRangeKernel, ptr align 8 %_arg_UserRange, i64 16, i1 false)
  %KernelFunc = getelementptr inbounds i8, ptr %RoundedRangeKernel, i64 16
  call void @llvm.lifetime.end.p0(i64 -1, ptr nonnull %RoundedRangeKernel)
  ret void
}

; CHECK: OpFunction
; CHECK: %[[#TestVar:]] = OpVariable
; CHECK: OpLifetimeStart %[[#TestVar]], 1
; CHECK: OpCopyMemorySized
; CHECK: OpInBoundsPtrAccessChain
; CHECK: OpLifetimeStop %[[#TestVar]], 1
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
