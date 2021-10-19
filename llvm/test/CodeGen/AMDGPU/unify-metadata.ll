; RUN: opt -mtriple=amdgcn--amdhsa -amdgpu-unify-metadata -S < %s | FileCheck -check-prefix=ALL %s
; RUN: opt -mtriple=amdgcn--amdhsa -passes=amdgpu-unify-metadata -S < %s | FileCheck -check-prefix=ALL %s

; This test check that we have a singe metadata value after linking several
; modules for records such as opencl.ocl.version, llvm.ident and similar.

; ALL-DAG: !opencl.ocl.version = !{![[OCL_VER:[0-9]+]]}
; ALL-DAG: !llvm.ident = !{![[LLVM_IDENT_0:[0-9]+]], ![[LLVM_IDENT_1:[0-9]+]]}
; ALL-DAG: !opencl.used.extensions = !{![[USED_EXT_0:[0-9]+]], ![[USED_EXT_1:[0-9]+]], ![[USED_EXT_2:[0-9]+]]}

; ALL-DAG: ![[OCL_VER]] = !{i32 1, i32 2}
; ALL-DAG: ![[LLVM_IDENT_0]] = !{!"clang version 4.0"}
; ALL-DAG: ![[LLVM_IDENT_1]] = !{!"clang version 4.0 (rLXXXXXX)"}
; ALL-DAG: ![[USED_EXT_0]] = !{!"cl_images"}
; ALL-DAG: ![[USED_EXT_1]] = !{!"cl_khr_fp16"}
; ALL-DAG: ![[USED_EXT_2]] = !{!"cl_doubles"}

!opencl.ocl.version = !{!1, !0, !0, !0}
!llvm.ident = !{!2, !2, !2, !2, !6}
!opencl.used.extensions = !{!3, !3, !4, !5}

!0 = !{i32 2, i32 0}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 4.0"}
!3 = !{!"cl_images", !"cl_khr_fp16"}
!4 = !{!"cl_images", !"cl_doubles"}
!5 = !{}
!6 = !{!"clang version 4.0 (rLXXXXXX)"}
