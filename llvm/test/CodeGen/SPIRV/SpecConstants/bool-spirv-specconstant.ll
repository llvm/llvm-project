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
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  %add.ptr.i = getelementptr inbounds i8, i8 addrspace(1)* %_arg_, i64 %2
  %3 = call i1 @_Z20__spirv_SpecConstantia(i32 0, i8 1)
  %ptridx.ascast.i.i = addrspacecast i8 addrspace(1)* %add.ptr.i to i8 addrspace(4)*
  %selected = select i1 %3, i8 0, i8 1
  %frombool.i = zext i1 %3 to i8
  %sum = add i8 %frombool.i, %selected
  store i8 %selected, i8 addrspace(4)* %ptridx.ascast.i.i, align 1
  ret void
}

declare i1 @_Z20__spirv_SpecConstantia(i32, i8)
