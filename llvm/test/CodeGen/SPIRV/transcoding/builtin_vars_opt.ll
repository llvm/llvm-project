; RUN: llc -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s --check-prefix=CHECK-SPIRV

;; The IR was generated from the following source:
;; #include <CL/sycl.hpp>
;;
;; template <typename T, int N>
;; class sycl_subgr;
;;
;; using namespace cl::sycl;
;;
;; int main() {
;;   queue Queue;
;;   int X = 8;
;;   nd_range<1> NdRange(X, X);
;;   buffer<int> buf(X);
;;   Queue.submit([&](handler &cgh) {
;;     auto acc = buf.template get_access<access::mode::read_write>(cgh);
;;     cgh.parallel_for<sycl_subgr<int, 0>>(NdRange, [=](nd_item<1> NdItem) {
;;       intel::sub_group SG = NdItem.get_sub_group();
;;       if (X % 2) {
;;         acc[0] = SG.get_max_local_range()[0];
;;       }
;;       acc[1] = (X % 3) ? 1 : SG.get_max_local_range()[0];
;;     });
;;   });
;;   return 0;
;; }
;; Command line:
;; clang -fsycl -fsycl-device-only -Xclang -fsycl-enable-optimizations tmp.cpp -o tmp.bc
;; llvm-spirv tmp.bc -s -o builtin_vars_opt.ll

; CHECK-SPIRV-DAG: OpDecorate %[[#SG_MaxSize_BI:]] BuiltIn SubgroupMaxSize
; CHECK-SPIRV-DAG: OpDecorate %[[#SG_MaxSize_BI:]] Constant
; CHECK-SPIRV-DAG: OpDecorate %[[#SG_MaxSize_BI:]] LinkageAttributes "__spirv_BuiltInSubgroupMaxSize" Import

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTS10sycl_subgrIiLi0EE = comdat any

@__spirv_BuiltInSubgroupMaxSize = external dso_local local_unnamed_addr addrspace(1) constant i32, align 4


define weak_odr dso_local spir_kernel void @_ZTS10sycl_subgrIiLi0EE(i32 %_arg_, i32 addrspace(1)* %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_4, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_5) local_unnamed_addr comdat {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_5, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_1, i64 %1
  %2 = and i32 %_arg_, 1
  %tobool.not.i = icmp eq i32 %2, 0
  %3 = addrspacecast i32 addrspace(1)* @__spirv_BuiltInSubgroupMaxSize to i32 addrspace(4)*
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %entry
  %4 = load i32, i32 addrspace(4)* %3, align 4
  %ptridx.ascast.i14.i = addrspacecast i32 addrspace(1)* %add.ptr.i to i32 addrspace(4)*
  store i32 %4, i32 addrspace(4)* %ptridx.ascast.i14.i, align 4
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %entry
  %rem3.i = srem i32 %_arg_, 3
  %tobool4.not.i = icmp eq i32 %rem3.i, 0
  br i1 %tobool4.not.i, label %cond.false.i, label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_7nd_itemILi1EEEE_clES5_.exit"

cond.false.i:                                     ; preds = %if.end.i
  %5 = load i32, i32 addrspace(4)* %3, align 4
  br label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_7nd_itemILi1EEEE_clES5_.exit"

"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_7nd_itemILi1EEEE_clES5_.exit": ; preds = %cond.false.i, %if.end.i
  %cond.i = phi i32 [ %5, %cond.false.i ], [ 1, %if.end.i ]
  %ptridx.i.i = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr.i, i64 1
  %ptridx.ascast.i.i = addrspacecast i32 addrspace(1)* %ptridx.i.i to i32 addrspace(4)*
  store i32 %cond.i, i32 addrspace(4)* %ptridx.ascast.i.i, align 4
  ret void
}
