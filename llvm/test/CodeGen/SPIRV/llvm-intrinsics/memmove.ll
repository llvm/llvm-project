; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK-NOT: llvm.memmove

; CHECK-DAG: %[[#TYPEINT:]] = OpTypeInt 32 0
; CHECK-DAG: %[[#I8:]] = OpTypeInt 8 0
; CHECK-DAG: %[[#I64:]] = OpTypeInt 64 0
; CHECK-DAG: %[[#C128:]] = OpConstant %[[#TYPEINT]] 128
; CHECK-DAG: %[[#C68:]] = OpConstant %[[#TYPEINT]] 68
; CHECK-DAG: %[[#C72:]] = OpConstant %[[#TYPEINT]] 72
; CHECK-DAG: %[[#C32:]] = OpConstant %[[#I64]] 32
; CHECK-DAG: %[[#I8CrossWG_PTR:]] = OpTypePointer CrossWorkgroup %[[#I8]]
; CHECK-DAG: %[[#I8Generic_PTR:]] = OpTypePointer Generic %[[#I8]]

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#ARG_IN1:]] = OpFunctionParameter %[[#I8CrossWG_PTR]]
; CHECK: %[[#ARG_OUT1:]] = OpFunctionParameter %[[#I8CrossWG_PTR]]
; CHECK: %[[#CONVERT1:]] = OpUConvert %[[#I64]] %[[#C128]]
; CHECK: OpCopyMemorySized %[[#ARG_OUT1]] %[[#ARG_IN1]] %[[#CONVERT1]] Aligned 64

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#ARG_IN2:]] = OpFunctionParameter %[[#I8CrossWG_PTR]]
; CHECK: %[[#FP:]] = OpFunctionParameter %[[#I8Generic_PTR]]
; CHECK: %[[#ARG_OUT2:]] = OpGenericCastToPtr %[[#I8CrossWG_PTR]] %[[#FP:]]
; CHECK: %[[#CONVERT2:]] = OpUConvert %[[#I64]] %[[#C68]]
; CHECK: CopyMemorySized %[[#ARG_OUT2]] %[[#ARG_IN2]] %[[#CONVERT2]] Aligned 64

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#ARG_IN3:]] = OpFunctionParameter %[[#I8CrossWG_PTR]]
; CHECK: %[[#ARG_OUT3:]] = OpFunctionParameter %[[#I8CrossWG_PTR]]
; CHECK: %[[#CONVERT3:]] = OpUConvert %[[#I64]] %[[#C72]]
; CHECK: OpCopyMemorySized %[[#ARG_OUT3]] %[[#ARG_IN3]] %[[#CONVERT3]] Aligned 1

; CHECK: %[[#]] = OpFunction
; CHECK: %[[#PHI:]] = OpPhi %[[#]] %[[#]] %[[#]] %[[#]] %[[#]]
; CHECK: %[[#CASTTOGENERIC:]] = OpPtrCastToGeneric %[[#]] %[[#]]
; CHECK: OpCopyMemorySized %[[#CASTTOGENERIC]] %[[#PHI]] %[[#C32]] Aligned 8

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir-unknown-unknown"

%struct.SomeStruct = type { <16 x float>, i32, [60 x i8] }
%class.kfunc = type <{ i32, i32, i32, [4 x i8] }>

@InvocIndex = external local_unnamed_addr addrspace(1) constant i64, align 8
@"func_object1" = internal addrspace(3) global %class.kfunc zeroinitializer, align 8

; Function Attrs: nounwind
define spir_kernel void @test_full_move(%struct.SomeStruct addrspace(1)* captures(none) readonly %in, %struct.SomeStruct addrspace(1)* captures(none) %out) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
  %1 = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
  %2 = bitcast %struct.SomeStruct addrspace(1)* %out to i8 addrspace(1)*
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* align 64 %2, i8 addrspace(1)* align 64 %1, i32 128, i1 false)
  ret void
}

define spir_kernel void @test_partial_move(%struct.SomeStruct addrspace(1)* captures(none) readonly %in, %struct.SomeStruct addrspace(4)* captures(none) %out) {
  %1 = bitcast %struct.SomeStruct addrspace(1)* %in to i8 addrspace(1)*
  %2 = bitcast %struct.SomeStruct addrspace(4)* %out to i8 addrspace(4)*
  %3 = addrspacecast i8 addrspace(4)* %2 to i8 addrspace(1)*
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* align 64 %3, i8 addrspace(1)* align 64 %1, i32 68, i1 false)
  ret void
}

; Function Attrs: nounwind
define spir_kernel void @test_array(i8 addrspace(1)* %in, i8 addrspace(1)* %out) {
  call void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* %out, i8 addrspace(1)* %in, i32 72, i1 false)
  ret void
}

; Function Attrs: convergent norecurse
define weak_odr dso_local spir_kernel void @test_phi() local_unnamed_addr {
entry:
  %0 = alloca i32, align 8
  %1 = addrspacecast i32* %0 to i32 addrspace(4)*
  %2 = load i64, i64 addrspace(1)* @InvocIndex, align 8
  %cmp = icmp eq i64 %2, 0
  br i1 %cmp, label %leader, label %entry.merge_crit_edge

entry.merge_crit_edge:                            ; preds = %entry
  %3 = bitcast i32 addrspace(4)* %1 to i8 addrspace(4)*
  br label %merge

leader:                                           ; preds = %entry
  %4 = bitcast i32 addrspace(4)* %1 to i8 addrspace(4)*
  br label %merge

merge:                                            ; preds = %entry.merge_crit_edge, %leader
  %phi = phi i8 addrspace(4)* [ %3, %entry.merge_crit_edge ], [ %4, %leader ]
  %5 = addrspacecast i8 addrspace(3)* bitcast (%class.kfunc addrspace(3)* @"func_object1" to i8 addrspace(3)*) to i8 addrspace(4)*
  call void @llvm.memmove.p4i8.p4i8.i64(i8 addrspace(4)* align 8 dereferenceable(32) %5, i8 addrspace(4)* align 8 dereferenceable(32) %phi, i64 32, i1 false)
  ret void
}

; Function Attrs: argmemonly nofree nosync nounwind willreturn
declare void @llvm.memmove.p4i8.p4i8.i64(i8 addrspace(4)* captures(none) writeonly, i8 addrspace(4)* captures(none) readonly, i64, i1 immarg)

; Function Attrs: nounwind
declare void @llvm.memmove.p1i8.p1i8.i32(i8 addrspace(1)* captures(none), i8 addrspace(1)* captures(none) readonly, i32, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no_infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}

!1 = !{i32 1, i32 1}
!2 = !{!"none", !"none"}
!3 = !{!"struct SomeStruct*", !"struct SomeStruct*"}
!4 = !{!"struct SomeStruct*", !"struct SomeStruct*"}
!5 = !{!"const", !""}
!7 = !{i32 1, i32 2}
!8 = !{}
