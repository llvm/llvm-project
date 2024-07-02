; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpFunction
; CHECK: %[[FooArg:.*]] = OpVariable
; CHECK: OpLifetimeStart %[[FooArg]], 0
; CHECK: OpCopyMemorySized
; CHECK: OpBitcast
; CHECK: OpInBoundsPtrAccessChain
; CHECK: OpLifetimeStop %[[FooArg]], 0

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
