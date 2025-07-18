; RUN: llc -verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=CHECK-KERNEL
; RUN: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-KERNEL: OpCapability Kernel
; CHECK-KERNEL-NOT: OpCapability Shader 
; CHECK-KERNEL: OpTypeArray
; CHECK-KERNEL-NOT: OpTypeRuntimeArray

%"class.sycl::_V1::detail::half_impl::half" = type { half }

; Function Attrs: mustprogress norecurse nounwind
define spir_kernel void @foo(ptr addrspace(3) noundef align 2 %_arg_temp, ptr addrspace(1) noundef align 2 %_arg_acc_a){
entry:
  %0 = getelementptr %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %_arg_acc_a, i64 15 
  %add.ptr.i = getelementptr %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %0, i64 10 
  %4 = getelementptr %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %add.ptr.i, i64 20 
  %arrayidx.i5.i = getelementptr %"class.sycl::_V1::detail::half_impl::half", ptr addrspace(1) %4, i64 35
  %arrayidx7.i = getelementptr inbounds [0 x [32 x %"class.sycl::_V1::detail::half_impl::half"]], ptr addrspace(3) %_arg_temp, i64 1, i64 25, i64 30
  %5 = load i16, ptr addrspace(1) %arrayidx.i5.i, align 2
  store i16 %5, ptr addrspace(3) %arrayidx7.i, align 2
  ret void
}
