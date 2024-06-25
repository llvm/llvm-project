; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; TODO: llc -O0 -mtriple=spirv32-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-SPIRV: OpName %[[#Var:]] "var"
; CHECK-SPIRV: OpName %[[#GVar:]] "g_var"
; CHECK-SPIRV: OpName %[[#AVar:]] "a_var"
; CHECK-SPIRV: OpName %[[#PVar:]] "p_var"

; CHECK-SPIRV-DAG: %[[#CharTy:]] = OpTypeInt 8 0
; CHECK-SPIRV-DAG: %[[#IntTy:]] = OpTypeInt 32 0
; CHECK-SPIRV-DAG: %[[#V2CharTy:]] = OpTypeVector %[[#CharTy]] 2
; CHECK-SPIRV-DAG: %[[#V2ConstNull:]] = OpConstantNull %[[#V2CharTy]]
; CHECK-SPIRV-DAG: %[[#Const1:]] = OpConstant %[[#CharTy]] 1
; CHECK-SPIRV-DAG: %[[#Const2:]] = OpConstant %[[#IntTy]] 2
; CHECK-SPIRV-DAG: %[[#Arr2V2CharTy:]] = OpTypeArray %[[#V2CharTy]] %[[#Const2]]
; CHECK-SPIRV-DAG: %[[#LongTy:]] = OpTypeInt 64 0
; CHECK-SPIRV-DAG: %[[#PtrV2CharTy:]] = OpTypePointer CrossWorkgroup %[[#V2CharTy]]
; CHECK-SPIRV-DAG: %[[#V2Char1:]] = OpConstantComposite %[[#V2CharTy]] %[[#Const1]] %[[#Const1]]
; CHECK-SPIRV-DAG: %[[#Arr2V2Char:]] = OpConstantComposite %[[#Arr2V2CharTy]] %[[#V2Char1]] %[[#V2Char1]]
; CHECK-SPIRV-DAG: %[[#PtrCharTy:]] = OpTypePointer CrossWorkgroup %[[#CharTy]]
; CHECK-SPIRV-DAG: %[[#PtrArr2V2CharTy:]] = OpTypePointer CrossWorkgroup %[[#Arr2V2CharTy]]
; CHECK-SPIRV-DAG: %[[#IntZero:]] = OpConstantNull %[[#IntTy]]
; CHECK-SPIRV-DAG: %[[#LongZero:]] = OpConstantNull %[[#LongTy]]
; CHECK-SPIRV-DAG: %[[#ConstLong2:]] = OpConstant %[[#LongTy]] 2
; CHECK-SPIRV-DAG: %[[#PvarInit:]] = OpSpecConstantOp %[[#PtrCharTy]] 70 %[[#VarV2Char:]] %[[#IntZero]] %[[#ConstLong2]]
; CHECK-SPIRV-DAG: %[[#PtrPtrCharTy:]] = OpTypePointer CrossWorkgroup %[[#PtrCharTy]]
; CHECK-SPIRV-DAG: %[[#AVar]] = OpVariable %[[#PtrArr2V2CharTy]] CrossWorkgroup %[[#Arr2V2Char]]
; CHECK-SPIRV-DAG: %[[#PVar]] = OpVariable %[[#PtrPtrCharTy]] CrossWorkgroup %[[#PvarInit]]
; CHECK-SPIRV-DAG: %[[#GVar]] = OpVariable %[[#PtrV2CharTy]] CrossWorkgroup %[[#V2Char1]]
; CHECK-SPIRV-DAG: %[[#]] = OpVariable %[[#PtrV2CharTy]] CrossWorkgroup %[[#V2ConstNull]]

; As an option: %[[#Const0:]] = OpConstant %[[#CharTy]] 0
;               %[[#V2CharZero:]] = OpConstantComposite %[[#V2CharTy]] %[[#Const0]] %[[#Const0]]
;               %[[#]] = OpVariable %[[#PtrV2CharTy]] CrossWorkgroup %[[#V2CharZero]]

@var = addrspace(1) global <2 x i8> zeroinitializer, align 2
@g_var = addrspace(1) global <2 x i8> <i8 1, i8 1>, align 2
@a_var = addrspace(1) global [2 x <2 x i8>] [<2 x i8> <i8 1, i8 1>, <2 x i8> <i8 1, i8 1>], align 2
@p_var = addrspace(1) global ptr addrspace(1) getelementptr (i8, ptr addrspace(1) @a_var, i64 2), align 8

define spir_func <2 x i8> @from_buf(<2 x i8> %a) #0 {
entry:
  ret <2 x i8> %a
}

define spir_func <2 x i8> @to_buf(<2 x i8> %a) #0 {
entry:
  ret <2 x i8> %a
}

define spir_kernel void @writer(ptr addrspace(1) %src, i32 %idx) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !7 !kernel_arg_type_qual !8 !kernel_arg_base_type !7 !spirv.ParameterDecorations !9 {
entry:
  %arrayidx = getelementptr inbounds <2 x i8>, ptr addrspace(1) %src, i64 0
  %0 = load <2 x i8>, ptr addrspace(1) %arrayidx, align 2
  %call = call spir_func <2 x i8> @from_buf(<2 x i8> %0) #0
  store <2 x i8> %call, ptr addrspace(1) @var, align 2
  %arrayidx1 = getelementptr inbounds <2 x i8>, ptr addrspace(1) %src, i64 1
  %1 = load <2 x i8>, ptr addrspace(1) %arrayidx1, align 2
  %call2 = call spir_func <2 x i8> @from_buf(<2 x i8> %1) #0
  store <2 x i8> %call2, ptr addrspace(1) @g_var, align 2
  %arrayidx3 = getelementptr inbounds <2 x i8>, ptr addrspace(1) %src, i64 2
  %2 = load <2 x i8>, ptr addrspace(1) %arrayidx3, align 2
  %call4 = call spir_func <2 x i8> @from_buf(<2 x i8> %2) #0
  %3 = getelementptr inbounds [2 x <2 x i8>], ptr addrspace(1) @a_var, i64 0, i64 0
  store <2 x i8> %call4, ptr addrspace(1) %3, align 2
  %arrayidx5 = getelementptr inbounds <2 x i8>, ptr addrspace(1) %src, i64 3
  %4 = load <2 x i8>, ptr addrspace(1) %arrayidx5, align 2
  %call6 = call spir_func <2 x i8> @from_buf(<2 x i8> %4) #0
  %5 = getelementptr inbounds [2 x <2 x i8>], ptr addrspace(1) @a_var, i64 0, i64 1
  store <2 x i8> %call6, ptr addrspace(1) %5, align 2
  %idx.ext = zext i32 %idx to i64
  %add.ptr = getelementptr inbounds <2 x i8>, ptr addrspace(1) %3, i64 %idx.ext
  store ptr addrspace(1) %add.ptr, ptr addrspace(1) @p_var, align 8
  ret void
}

define spir_kernel void @reader(ptr addrspace(1) %dest, <2 x i8> %ptr_write_val) #0 !kernel_arg_addr_space !5 !kernel_arg_access_qual !6 !kernel_arg_type !10 !kernel_arg_type_qual !8 !kernel_arg_base_type !10 !spirv.ParameterDecorations !9 {
entry:
  %call = call spir_func <2 x i8> @from_buf(<2 x i8> %ptr_write_val) #0
  %0 = load ptr addrspace(1), ptr addrspace(1) @p_var, align 8
  store volatile <2 x i8> %call, ptr addrspace(1) %0, align 2
  %1 = load <2 x i8>, ptr addrspace(1) @var, align 2
  %call1 = call spir_func <2 x i8> @to_buf(<2 x i8> %1) #0
  %arrayidx = getelementptr inbounds <2 x i8>, ptr addrspace(1) %dest, i64 0
  store <2 x i8> %call1, ptr addrspace(1) %arrayidx, align 2
  %2 = load <2 x i8>, ptr addrspace(1) @g_var, align 2
  %call2 = call spir_func <2 x i8> @to_buf(<2 x i8> %2) #0
  %arrayidx3 = getelementptr inbounds <2 x i8>, ptr addrspace(1) %dest, i64 1
  store <2 x i8> %call2, ptr addrspace(1) %arrayidx3, align 2
  %3 = getelementptr inbounds [2 x <2 x i8>], ptr addrspace(1) @a_var, i64 0, i64 0
  %4 = load <2 x i8>, ptr addrspace(1) %3, align 2
  %call4 = call spir_func <2 x i8> @to_buf(<2 x i8> %4) #0
  %arrayidx5 = getelementptr inbounds <2 x i8>, ptr addrspace(1) %dest, i64 2
  store <2 x i8> %call4, ptr addrspace(1) %arrayidx5, align 2
  %5 = getelementptr inbounds [2 x <2 x i8>], ptr addrspace(1) @a_var, i64 0, i64 1
  %6 = load <2 x i8>, ptr addrspace(1) %5, align 2
  %call6 = call spir_func <2 x i8> @to_buf(<2 x i8> %6) #0
  %arrayidx7 = getelementptr inbounds <2 x i8>, ptr addrspace(1) %dest, i64 3
  store <2 x i8> %call6, ptr addrspace(1) %arrayidx7, align 2
  ret void
}

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!2}
!opencl.ocl.version = !{!2}
!opencl.used.extensions = !{!3}
!opencl.used.optional.core.features = !{!3}
!spirv.Generator = !{!4}

!0 = !{i32 2, i32 2}
!1 = !{i32 3, i32 200000}
!2 = !{i32 2, i32 0}
!3 = !{}
!4 = !{i16 6, i16 14}
!5 = !{i32 1, i32 0}
!6 = !{!"none", !"none"}
!7 = !{!"char2*", !"int"}
!8 = !{!"", !""}
!9 = !{!3, !3}
!10 = !{!"char2*", !"char2"}
