; RUN: llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s

; CHECK: OpDecorate %[[#SpecConst:]] SpecId 0
; CHECK: %[[#SpecConst]] = OpSpecConstant %[[#]] 70
; CHECK: %[[#]] = OpPhi %[[#]] %[[#]] %[[#]] %[[#SpecConst]] %[[#]]

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }

$_ZTS6kernel = comdat any

define weak_odr dso_local spir_kernel void @_ZTS6kernel(i8 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3) local_unnamed_addr comdat {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %1 = addrspacecast i64* %0 to i64 addrspace(4)*
  %2 = load i64, i64 addrspace(4)* %1, align 8
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.body.i.i, %entry
  %value.0.i.i = phi i8 [ -1, %entry ], [ %3, %for.body.i.i ]
  %cmp.i.i = phi i1 [ true, %entry ], [ false, %for.body.i.i ]
  br i1 %cmp.i.i, label %for.body.i.i, label %_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlNS0_14kernel_handlerEE_clES4_.exit

for.body.i.i:                                     ; preds = %for.cond.i.i
  %3 = call i8 @_Z20__spirv_SpecConstantia(i32 0, i8 70)
  br label %for.cond.i.i

_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlNS0_14kernel_handlerEE_clES4_.exit: ; preds = %for.cond.i.i
  %add.ptr.i = getelementptr inbounds i8, i8 addrspace(1)* %_arg_, i64 %2
  %arrayidx.ascast.i.i = addrspacecast i8 addrspace(1)* %add.ptr.i to i8 addrspace(4)*
  store i8 %value.0.i.i, i8 addrspace(4)* %arrayidx.ascast.i.i, align 1
  ret void
}

declare i8 @_Z20__spirv_SpecConstantia(i32, i8)
