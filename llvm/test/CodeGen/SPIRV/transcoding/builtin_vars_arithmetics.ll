; RUN: llc -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

;; The IR was generated from the following source:
;; #include <CL/sycl.hpp>
;;
;; int main() {
;;   sycl::queue Queue;
;;   int array[2][3] = {0};
;;   {
;;     sycl::range<2> Range(2, 3);
;;     sycl::buffer<int, 2> buf((int *)array, Range,
;;                              {cl::sycl::property::buffer::use_host_ptr()});
;;
;;     Queue.submit([&](sycl::handler &cgh) {
;;       auto acc = buf.get_access<sycl::access::mode::read_write>(cgh);
;;       cgh.parallel_for<class dim2_subscr>(Range, [=](sycl::item<2> itemID) {
;;         acc[itemID.get_id(0)][itemID.get_id(1)] += itemID.get_linear_id();
;;       });
;;     });
;;     Queue.wait();
;;   }
;;   return 0;
;; }
;; Command line:
;; clang++ -fsycl -fsycl-device-only emit-llvm tmp.cpp -o tmp.bc
;; llvm-spirv tmp.bc -spirv-text -o builtin_vars_arithmetics.ll

; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalInvocationId:]] BuiltIn GlobalInvocationId
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalSize:]] BuiltIn GlobalSize
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalOffset:]] BuiltIn GlobalOffset
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalInvocationId]] Constant
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalSize]] Constant
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalOffset]] Constant
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalOffset]] LinkageAttributes "__spirv_BuiltInGlobalOffset" Import 
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalSize]] LinkageAttributes "__spirv_BuiltInGlobalSize" Import 
; CHECK-SPIRV-DAG: OpDecorate %[[#GlobalInvocationId]] LinkageAttributes "__spirv_BuiltInGlobalInvocationId" Import 

%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" = type { [2 x i64] }
%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi2EEE.cl::sycl::detail::array" }

$"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11dim2_subscr" = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalOffset = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

define weak_odr dso_local spir_kernel void @"_ZTSZZ4mainENK3$_0clERN2cl4sycl7handlerEE11dim2_subscr"(i32 addrspace(1)* %_arg_, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_1, %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range") align 8 %_arg_2, %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr comdat {
entry:
  %agg.tmp4.sroa.0.sroa.2.0.agg.tmp4.sroa.0.0..sroa_cast.sroa_idx65 = getelementptr inbounds %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range", %"class._ZTSN2cl4sycl5rangeILi2EEE.cl::sycl::range"* %_arg_2, i64 0, i32 0, i32 0, i64 1
  %agg.tmp4.sroa.0.sroa.2.0.copyload = load i64, i64* %agg.tmp4.sroa.0.sroa.2.0.agg.tmp4.sroa.0.0..sroa_cast.sroa_idx65, align 8
  %agg.tmp5.sroa.0.sroa.0.0.agg.tmp5.sroa.0.0..sroa_cast.sroa_idx = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 0
  %agg.tmp5.sroa.0.sroa.0.0.copyload = load i64, i64* %agg.tmp5.sroa.0.sroa.0.0.agg.tmp5.sroa.0.0..sroa_cast.sroa_idx, align 8
  %agg.tmp5.sroa.0.sroa.2.0.agg.tmp5.sroa.0.0..sroa_cast.sroa_idx69 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi2EEE.cl::sycl::id"* %_arg_3, i64 0, i32 0, i32 0, i64 1
  %agg.tmp5.sroa.0.sroa.2.0.copyload = load i64, i64* %agg.tmp5.sroa.0.sroa.2.0.agg.tmp5.sroa.0.0..sroa_cast.sroa_idx69, align 8
  %0 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId to <3 x i64> addrspace(4)*), align 32
  %1 = extractelement <3 x i64> %0, i64 1
  %2 = extractelement <3 x i64> %0, i64 0
  %3 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalSize to <3 x i64> addrspace(4)*), align 32
  %4 = load <3 x i64>, <3 x i64> addrspace(4)* addrspacecast (<3 x i64> addrspace(1)* @__spirv_BuiltInGlobalOffset to <3 x i64> addrspace(4)*), align 32
  %5 = sub <3 x i64> %0, %4
  %6 = sub <3 x i64> %0, %4
  %7 = extractelement <3 x i64> %6, i64 0
  %8 = extractelement <3 x i64> %5, i32 1
  %9 = extractelement <3 x i64> %3, i64 0
  %10 = mul i64 %8, %9
  %add.i.i.i = add i64 %7, %10
  %add6.i.i.i.i = add i64 %1, %agg.tmp5.sroa.0.sroa.0.0.copyload
  %mul.1.i.i.i.i = mul i64 %add6.i.i.i.i, %agg.tmp4.sroa.0.sroa.2.0.copyload
  %add.1.i.i.i.i = add i64 %2, %agg.tmp5.sroa.0.sroa.2.0.copyload
  %add6.1.i.i.i.i = add i64 %add.1.i.i.i.i, %mul.1.i.i.i.i
  %ptridx.i.i.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_, i64 %add6.1.i.i.i.i
  %ptridx.ascast.i.i.i = addrspacecast i32 addrspace(1)* %ptridx.i.i.i to i32 addrspace(4)*
  %11 = load i32, i32 addrspace(4)* %ptridx.ascast.i.i.i, align 4
  %12 = trunc i64 %add.i.i.i to i32
  %conv5.i = add i32 %11, %12
  store i32 %conv5.i, i32 addrspace(4)* %ptridx.ascast.i.i.i, align 4
  ret void
}
