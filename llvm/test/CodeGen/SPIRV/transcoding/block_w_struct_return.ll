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

%struct.__opencl_block_literal_generic = type { i32, i32, i8 addrspace(4)* }
%struct.A = type { i32 }

@__block_literal_global = internal addrspace(1) constant { i32, i32, i8 addrspace(4)* } { i32 12, i32 4, i8 addrspace(4)* addrspacecast (i8* bitcast (void (%struct.A*, i8 addrspace(4)*, %struct.A*)* @__block_ret_struct_block_invoke to i8*) to i8 addrspace(4)*) }, align 4

define dso_local spir_kernel void @block_ret_struct(i32 addrspace(1)* noundef %res) {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 4
  %kernelBlock = alloca %struct.__opencl_block_literal_generic addrspace(4)*, align 4
  %tid = alloca i32, align 4
  %aa = alloca %struct.A, align 4
  %tmp = alloca %struct.A, align 4
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 4
  %0 = bitcast %struct.__opencl_block_literal_generic addrspace(4)** %kernelBlock to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %0)
  store %struct.__opencl_block_literal_generic addrspace(4)* addrspacecast (%struct.__opencl_block_literal_generic addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to %struct.__opencl_block_literal_generic addrspace(1)*) to %struct.__opencl_block_literal_generic addrspace(4)*), %struct.__opencl_block_literal_generic addrspace(4)** %kernelBlock, align 4
  %1 = bitcast i32* %tid to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %1)
  %call = call spir_func i32 @_Z13get_global_idj(i32 noundef 0)
  store i32 %call, i32* %tid, align 4
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 4
  %3 = load i32, i32* %tid, align 4
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %2, i32 %3
  store i32 -1, i32 addrspace(1)* %arrayidx, align 4
  %4 = bitcast %struct.A* %aa to i8*
  call void @llvm.lifetime.start.p0i8(i64 4, i8* %4)
  %a = getelementptr inbounds %struct.A, %struct.A* %aa, i32 0, i32 0
  store i32 5, i32* %a, align 4
  call spir_func void @__block_ret_struct_block_invoke(%struct.A* sret(%struct.A) align 4 %tmp, i8 addrspace(4)* noundef addrspacecast (i8 addrspace(1)* bitcast ({ i32, i32, i8 addrspace(4)* } addrspace(1)* @__block_literal_global to i8 addrspace(1)*) to i8 addrspace(4)*), %struct.A* noundef byval(%struct.A) align 4 %aa)
  %a1 = getelementptr inbounds %struct.A, %struct.A* %tmp, i32 0, i32 0
  %5 = load i32, i32* %a1, align 4
  %sub = sub nsw i32 %5, 6
  %6 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 4
  %7 = load i32, i32* %tid, align 4
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %6, i32 %7
  store i32 %sub, i32 addrspace(1)* %arrayidx2, align 4
  %8 = bitcast %struct.A* %aa to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %8)
  %9 = bitcast i32* %tid to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %9)
  %10 = bitcast %struct.__opencl_block_literal_generic addrspace(4)** %kernelBlock to i8*
  call void @llvm.lifetime.end.p0i8(i64 4, i8* %10)
  ret void
}

declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

define internal spir_func void @__block_ret_struct_block_invoke(%struct.A* noalias sret(%struct.A) align 4 %agg.result, i8 addrspace(4)* noundef %.block_descriptor, %struct.A* noundef byval(%struct.A) align 4 %a) {
entry:
  %.block_descriptor.addr = alloca i8 addrspace(4)*, align 4
  store i8 addrspace(4)* %.block_descriptor, i8 addrspace(4)** %.block_descriptor.addr, align 4
  %block = bitcast i8 addrspace(4)* %.block_descriptor to <{ i32, i32, i8 addrspace(4)* }> addrspace(4)*
  %a1 = getelementptr inbounds %struct.A, %struct.A* %a, i32 0, i32 0
  store i32 6, i32* %a1, align 4
  %0 = bitcast %struct.A* %agg.result to i8*
  %1 = bitcast %struct.A* %a to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 4 %0, i8* align 4 %1, i32 4, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i32, i1 immarg)

declare spir_func i32 @_Z13get_global_idj(i32 noundef)

declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)
