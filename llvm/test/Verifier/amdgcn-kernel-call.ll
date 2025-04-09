; RUN: not llvm-as < %s 2>&1 | FileCheck %s
; CHECK: Calls to kernel functions are not supported

target triple = "amdgcn"

@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @CalleeKernel() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local amdgpu_kernel void @CallerKernel() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  call amdgpu_kernel void @CalleeKernel() #1
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,256" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, i32 2}
!3 = !{!"clang version 21.0.0git (git@github.com:llvm/llvm-project.git 37565e60b3325324c9396ce236cb005a958f157e)"}
!4 = !{}
