; RUN: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV1_4
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; TODO(#60133): Requires updates following opaque pointer migration.
; XFAIL: *

;; TODO: We cannot check SPIR_V 1.1 and 1.4 simultaneously, implement additional
;;       run with CHECK-SPIRV1_1.

;; kernel void block_ret_struct(__global int* res)
;; {
;;   struct A {
;;       int a;
;;   };
;;   struct A (^kernelBlock)(struct A) = ^struct A(struct A a)
;;   {
;;     a.a = 6;
;;     return a;
;;   };
;;   size_t tid = get_global_id(0);
;;   res[tid] = -1;
;;   struct A aa;
;;   aa.a = 5;
;;   res[tid] = kernelBlock(aa).a - 6;
;; }

; CHECK-SPIRV1_4: OpEntryPoint Kernel %[[#]] "block_ret_struct" %[[#InterdaceId1:]] %[[#InterdaceId2:]]
; CHECK-SPIRV1_4: OpName %[[#InterdaceId1]] "__block_literal_global"
; CHECK-SPIRV1_4: OpName %[[#InterdaceId2]] "__spirv_BuiltInGlobalInvocationId"

; CHECK-SPIRV1_1: OpEntryPoint Kernel %[[#]] "block_ret_struct" %[[#InterdaceId1:]]
; CHECK-SPIRV1_1: OpName %[[#InterdaceId1]] "__spirv_BuiltInGlobalInvocationId"

; CHECK-SPIRV: OpName %[[#BlockInv:]] "__block_ret_struct_block_invoke"

; CHECK-SPIRV: %[[#IntTy:]] = OpTypeInt 32
; CHECK-SPIRV: %[[#Int8Ty:]] = OpTypeInt 8
; CHECK-SPIRV: %[[#Int8Ptr:]] = OpTypePointer Generic %[[#Int8Ty]]
; CHECK-SPIRV: %[[#StructTy:]] = OpTypeStruct %[[#IntTy]]{{$}}
; CHECK-SPIRV: %[[#StructPtrTy:]] = OpTypePointer Function %[[#StructTy]]

; CHECK-SPIRV: %[[#StructArg:]] = OpVariable %[[#StructPtrTy]] Function
; CHECK-SPIRV: %[[#StructRet:]] = OpVariable %[[#StructPtrTy]] Function
; CHECK-SPIRV: %[[#BlockLit:]] = OpPtrCastToGeneric %[[#Int8Ptr]] %[[#]]
; CHECK-SPIRV: %[[#]] = OpFunctionCall %[[#]] %[[#BlockInv]] %[[#StructRet]] %[[#BlockLit]] %[[#StructArg]]

%struct.__opencl_block_literal_generic = type { i32, i32, ptr addrspace(4) }
%struct.A = type { i32 }

@__block_literal_global = internal addrspace(1) constant { i32, i32, ptr addrspace(4) } { i32 12, i32 4, ptr addrspace(4) addrspacecast (ptr @__block_ret_struct_block_invoke to ptr addrspace(4)) }, align 4

define dso_local spir_kernel void @block_ret_struct(ptr addrspace(1) noundef %res) {
entry:
  %res.addr = alloca ptr addrspace(1), align 4
  %kernelBlock = alloca ptr addrspace(4), align 4
  %tid = alloca i32, align 4
  %aa = alloca %struct.A, align 4
  %tmp = alloca %struct.A, align 4
  store ptr addrspace(1) %res, ptr %res.addr, align 4
  %0 = bitcast ptr %kernelBlock to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %0)
  store ptr addrspace(4) addrspacecast (ptr addrspace(1) @__block_literal_global to ptr addrspace(4)), ptr %kernelBlock, align 4
  %1 = bitcast ptr %tid to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %1)
  %call = call spir_func i32 @_Z13get_global_idj(i32 noundef 0)
  store i32 %call, ptr %tid, align 4
  %2 = load ptr addrspace(1), ptr %res.addr, align 4
  %3 = load i32, ptr %tid, align 4
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %2, i32 %3
  store i32 -1, ptr addrspace(1) %arrayidx, align 4
  %4 = bitcast ptr %aa to ptr
  call void @llvm.lifetime.start.p0(i64 4, ptr %4)
  %a = getelementptr inbounds %struct.A, ptr %aa, i32 0, i32 0
  store i32 5, ptr %a, align 4
  call spir_func void @__block_ret_struct_block_invoke(ptr sret(%struct.A) align 4 %tmp, ptr addrspace(4) noundef addrspacecast (ptr addrspace(1) @__block_literal_global to ptr addrspace(4)), ptr noundef byval(%struct.A) align 4 %aa)
  %a1 = getelementptr inbounds %struct.A, ptr %tmp, i32 0, i32 0
  %5 = load i32, ptr %a1, align 4
  %sub = sub nsw i32 %5, 6
  %6 = load ptr addrspace(1), ptr %res.addr, align 4
  %7 = load i32, ptr %tid, align 4
  %arrayidx2 = getelementptr inbounds i32, ptr addrspace(1) %6, i32 %7
  store i32 %sub, ptr addrspace(1) %arrayidx2, align 4
  %8 = bitcast ptr %aa to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %8)
  %9 = bitcast ptr %tid to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %9)
  %10 = bitcast ptr %kernelBlock to ptr
  call void @llvm.lifetime.end.p0(i64 4, ptr %10)
  ret void
}

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

define internal spir_func void @__block_ret_struct_block_invoke(ptr noalias sret(%struct.A) align 4 %agg.result, ptr addrspace(4) noundef %.block_descriptor, ptr noundef byval(%struct.A) align 4 %a) {
entry:
  %.block_descriptor.addr = alloca ptr addrspace(4), align 4
  store ptr addrspace(4) %.block_descriptor, ptr %.block_descriptor.addr, align 4
  %block = bitcast ptr addrspace(4) %.block_descriptor to ptr addrspace(4)
  %a1 = getelementptr inbounds %struct.A, ptr %a, i32 0, i32 0
  store i32 6, ptr %a1, align 4
  %0 = bitcast ptr %agg.result to ptr
  %1 = bitcast ptr %a to ptr
  call void @llvm.memcpy.p0.p0.i32(ptr align 4 %0, ptr align 4 %1, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i32(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i32, i1 immarg)

declare spir_func i32 @_Z13get_global_idj(i32 noundef)

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)
