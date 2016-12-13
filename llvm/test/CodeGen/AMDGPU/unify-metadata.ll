; RUN: opt -mtriple=amdgcn--amdhsa -amdgpu-unify-metadata -S < %s | FileCheck -check-prefix=ALL %s

; This test check that we have a singe metadata value after linking several
; modules for records such as opencl.ocl.version, llvm.ident and similar.

; ALL-DAG: !opencl.ocl.version = !{![[OCL_VER:[0-9]+]]}
; ALL-DAG: !llvm.ident = !{![[LLVM_IDENT:[0-9]+]]}
; ALL-DAG: !opencl.used.extensions = !{![[USED_EXT:[0-9]+]]}
; ALL-DAG: ![[OCL_VER]] = !{i32 1, i32 2}
; ALL-DAG: ![[LLVM_IDENT]] = !{!"clang version 4.0 "}
; ALL-DAG: ![[USED_EXT]] = !{!"cl_images", !"cl_khr_fp16", !"cl_doubles"}

define void @test() {
   ret void
}

!opencl.ocl.version = !{!1, !0, !0, !0}
!llvm.ident = !{!2, !2, !2, !2}
!opencl.used.extensions = !{!3, !3, !4, !5}

!0 = !{i32 2, i32 0}
!1 = !{i32 1, i32 2}
!2 = !{!"clang version 4.0 "}
!3 = !{!"cl_images", !"cl_khr_fp16"}
!4 = !{!"cl_images", !"cl_doubles"}
!5 = !{}
