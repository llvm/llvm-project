; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64v1.2-unknown-unknown %s -o - -filetype=obj | spirv-val --target-env opencl2.2 %}

; CHECK: OpCapability Kernel
; CHECK-NOT: OpCapability Shader
; CHECK-DAG: %[[#float16:]] = OpTypeFloat 16
; CHECK-DAG: %[[#SyclHalfTy:]] = OpTypeStruct %[[#float16]]
; CHECK-DAG: %[[#i16:]] = OpTypeInt 16
; CHECK-DAG: %[[#i32:]] = OpTypeInt 32
; CHECK-DAG: %[[#i64:]] = OpTypeInt 64
; CHECK-DAG: %[[#ConstNull:]] = OpConstantNull %[[#i64]]
; CHECK-DAG: %[[#ConstOne:]] = OpConstant %[[#i64]] 1
; CHECK-DAG: %[[#ConstFive:]] = OpConstant %[[#i16]] 5
; CHECK-DAG: %[[#SyclHalfTyPtr:]] = OpTypePointer Function %[[#SyclHalfTy]]
; CHECK-DAG: %[[#i32Ptr:]] = OpTypePointer Function %[[#i32]]
; CHECK-DAG: %[[#StorePtrTy:]] = OpTypePointer Function %[[#i16]]

%"class.sycl::_V1::detail::half_impl::half" = type { half }

; Function Attrs: mustprogress norecurse nounwind
define spir_kernel void @foo(ptr %p){
; CHECK: OpFunction
; CHECK: %[[#Ptr:]] = OpFunctionParameter
; CHECK: OpLabel
; CHECK: %[[#BitcastOp:]] = OpInBoundsPtrAccessChain %[[#SyclHalfTyPtr]] %[[#Ptr]] %[[#ConstNull]] %[[#ConstNull]]
; CHECK: %[[#StorePtr:]] = OpBitcast %[[#StorePtrTy]] %[[#BitcastOp]]
; CHECK: OpStore %[[#StorePtr]] %[[#ConstFive]]
; CHECK: OpReturn
entry:
  %0 = getelementptr inbounds [0 x [32 x %"class.sycl::_V1::detail::half_impl::half"]], ptr %p, i64 0, i64 0, i64 0
  store i16 5, ptr %0
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define spir_kernel void @foo2(ptr %p){
; CHECK: OpFunction
; CHECK: %[[#Ptr:]] = OpFunctionParameter
; CHECK: OpLabel
; CHECK: %[[#BitcastOp:]] = OpInBoundsPtrAccessChain %[[#SyclHalfTyPtr]] %[[#Ptr]] %[[#ConstOne]] %[[#ConstOne]]
; CHECK: %[[#StorePtr:]] = OpBitcast %[[#StorePtrTy]] %[[#BitcastOp]]
; CHECK: OpStore %[[#StorePtr]] %[[#ConstFive]]
; CHECK: OpReturn
entry:
  %0 = getelementptr inbounds [0 x [32 x %"class.sycl::_V1::detail::half_impl::half"]], ptr %p, i64 0, i64 1, i64 1
  store i16 5, ptr %0
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define spir_kernel void @foo3(ptr %p){
; CHECK: OpFunction
; CHECK: %[[#Ptr:]] = OpFunctionParameter
; CHECK: OpLabel
; CHECK: %[[#BitcastOp:]] = OpInBoundsPtrAccessChain %[[#i32Ptr]] %[[#Ptr]] %[[#ConstNull]] %[[#ConstNull]]
; CHECK: %[[#StorePtr:]] = OpBitcast %[[#StorePtrTy]] %[[#BitcastOp]]
; CHECK: OpStore %[[#StorePtr]] %[[#ConstFive]]
; CHECK: OpReturn
entry:
  %0 = getelementptr inbounds [0 x [32 x i32]], ptr %p, i64 0, i64 0, i64 0
  store i16 5, ptr %0
  ret void
}

; Function Attrs: mustprogress norecurse nounwind
define spir_kernel void @foo4(ptr %p){
; CHECK: OpFunction
; CHECK: %[[#Ptr:]] = OpFunctionParameter
; CHECK: OpLabel
; CHECK: %[[#BitcastOp:]] = OpInBoundsPtrAccessChain %[[#i32Ptr]] %[[#Ptr]] %[[#ConstOne]] %[[#ConstOne]]
; CHECK: %[[#StorePtr:]] = OpBitcast %[[#StorePtrTy]] %[[#BitcastOp]]
; CHECK: OpStore %[[#StorePtr]] %[[#ConstFive]]
; CHECK: OpReturn
entry:
  %0 = getelementptr inbounds [0 x [32 x i32]], ptr %p, i64 0, i64 1, i64 1
  store i16 5, ptr %0
  ret void
}
