; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpDecorate %[[#BOOL_CONST:]] SpecId [[#]]
; CHECK: %[[#BOOL_TY:]] = OpTypeBool
; CHECK: %[[#BOOL_CONST]] = OpSpecConstantTrue %[[#BOOL_TY]]
; CHECK: %[[#]] = OpSelect %[[#]] %[[#BOOL_CONST]]
;; zext is also represented as Select because of how SPIR-V spec is written
; CHECK: %[[#]] = OpSelect %[[#]] %[[#BOOL_CONST]]

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel1" = comdat any

define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE7Kernel1"(i8 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr comdat {
entry:
  %gep1 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %cast1 = addrspacecast i64* %gep1 to i64 addrspace(4)*
  %load = load i64, i64 addrspace(4)* %cast1, align 8
  %gep2 = getelementptr inbounds i8, i8 addrspace(1)* %_arg_, i64 %load
  %call1 = call i1 @_Z20__spirv_SpecConstantia(i32 0, i8 1)
  %cast2 = addrspacecast i8 addrspace(1)* %gep2 to i8 addrspace(4)*
  %selected = select i1 %call1, i8 0, i8 1
  %bool = zext i1 %call1 to i8
  %sum = add i8 %bool, %selected
  store i8 %selected, i8 addrspace(4)* %cast2, align 1
  ret void
}

declare i1 @_Z20__spirv_SpecConstantia(i32, i8)
