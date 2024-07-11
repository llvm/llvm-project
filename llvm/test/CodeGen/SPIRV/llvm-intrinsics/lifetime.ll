; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: %[[#Char:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#PtrChar:]] = OpTypePointer Function %[[#Char]]
; CHECK: OpFunction
; CHECK: %[[#FooArg:]] = OpVariable
; CHECK: %[[#Casted1:]] = OpBitcast %[[#PtrChar]] %[[#FooArg]]
; CHECK: OpLifetimeStart %[[#Casted1]], 72
; CHECK: OpCopyMemorySized
; CHECK: OpBitcast
; CHECK: OpInBoundsPtrAccessChain
; CHECK: %[[#Casted2:]] = OpBitcast %[[#PtrChar]] %[[#FooArg]]
; CHECK: OpLifetimeStop %[[#Casted2]], 72

%tprange = type { %tparray }
%tparray = type { [2 x i64] }

define spir_func void @foo(ptr noundef byval(%tprange) align 8 %_arg_UserRange) {
  %RoundedRangeKernel = alloca %tprange, align 8
  call void @llvm.lifetime.start.p0(i64 72, ptr nonnull %RoundedRangeKernel) #7
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %RoundedRangeKernel, ptr align 8 %_arg_UserRange, i64 16, i1 false)
  %KernelFunc = getelementptr inbounds i8, ptr %RoundedRangeKernel, i64 16
  call void @llvm.lifetime.end.p0(i64 72, ptr nonnull %RoundedRangeKernel) #7
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
