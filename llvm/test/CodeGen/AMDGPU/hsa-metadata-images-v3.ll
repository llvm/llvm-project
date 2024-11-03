; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck --check-prefix=CHECK %s

%opencl.image1d_t = type opaque
%opencl.image1d_array_t = type opaque
%opencl.image1d_buffer_t = type opaque
%opencl.image2d_t = type opaque
%opencl.image2d_array_t = type opaque
%opencl.image2d_array_depth_t = type opaque
%opencl.image2d_array_msaa_t = type opaque
%opencl.image2d_array_msaa_depth_t = type opaque
%opencl.image2d_depth_t = type opaque
%opencl.image2d_msaa_t = type opaque
%opencl.image2d_msaa_depth_t = type opaque
%opencl.image3d_t = type opaque

; CHECK:         ---
; CHECK: amdhsa.kernels:
; CHECK:   - .args:
; CHECK:       - .name:           a
; CHECK:         .offset:         0
; CHECK:         .size:           8
; CHECK:         .type_name:      image1d_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           b
; CHECK:         .offset:         8
; CHECK:         .size:           8
; CHECK:         .type_name:      image1d_array_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           c
; CHECK:         .offset:         16
; CHECK:         .size:           8
; CHECK:         .type_name:      image1d_buffer_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           d
; CHECK:         .offset:         24
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           e
; CHECK:         .offset:         32
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           f
; CHECK:         .offset:         40
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_depth_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           g
; CHECK:         .offset:         48
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_msaa_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           h
; CHECK:         .offset:         56
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_msaa_depth_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           i
; CHECK:         .offset:         64
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_depth_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           j
; CHECK:         .offset:         72
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_msaa_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           k
; CHECK:         .offset:         80
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_msaa_depth_t
; CHECK:         .value_kind:     image
; CHECK:       - .name:           l
; CHECK:         .offset:         88
; CHECK:         .size:           8
; CHECK:         .type_name:      image3d_t
; CHECK:         .value_kind:     image
define amdgpu_kernel void @test(ptr addrspace(1) %a,
                                ptr addrspace(1) %b,
                                ptr addrspace(1) %c,
                                ptr addrspace(1) %d,
                                ptr addrspace(1) %e,
                                ptr addrspace(1) %f,
                                ptr addrspace(1) %g,
                                ptr addrspace(1) %h,
                                ptr addrspace(1) %i,
                                ptr addrspace(1) %j,
                                ptr addrspace(1) %k,
                                ptr addrspace(1) %l)
    !kernel_arg_type !1 !kernel_arg_base_type !1 {
  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 300}

!1 = !{!"image1d_t", !"image1d_array_t", !"image1d_buffer_t",
       !"image2d_t", !"image2d_array_t", !"image2d_array_depth_t",
       !"image2d_array_msaa_t", !"image2d_array_msaa_depth_t",
       !"image2d_depth_t", !"image2d_msaa_t", !"image2d_msaa_depth_t",
       !"image3d_t"}
