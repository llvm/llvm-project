; RUN: llc -mattr=+code-object-v3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mattr=+code-object-v3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX802 --check-prefix=NOTES %s
; RUN: llc -mattr=+code-object-v3 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

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
; CHECK:       - .address_space:  global
; CHECK:         .name:           a
; CHECK:         .offset:         0
; CHECK:         .size:           8
; CHECK:         .type_name:      image1d_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           b
; CHECK:         .offset:         8
; CHECK:         .size:           8
; CHECK:         .type_name:      image1d_array_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           c
; CHECK:         .offset:         16
; CHECK:         .size:           8
; CHECK:         .type_name:      image1d_buffer_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           d
; CHECK:         .offset:         24
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           e
; CHECK:         .offset:         32
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           f
; CHECK:         .offset:         40
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_depth_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           g
; CHECK:         .offset:         48
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_msaa_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           h
; CHECK:         .offset:         56
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_array_msaa_depth_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           i
; CHECK:         .offset:         64
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_depth_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           j
; CHECK:         .offset:         72
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_msaa_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           k
; CHECK:         .offset:         80
; CHECK:         .size:           8
; CHECK:         .type_name:      image2d_msaa_depth_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
; CHECK:       - .address_space:  global
; CHECK:         .name:           l
; CHECK:         .offset:         88
; CHECK:         .size:           8
; CHECK:         .type_name:      image3d_t
; CHECK:         .value_kind:     image
; CHECK:         .value_type:     struct
define amdgpu_kernel void @test(%opencl.image1d_t addrspace(1)* %a,
                                %opencl.image1d_array_t addrspace(1)* %b,
                                %opencl.image1d_buffer_t addrspace(1)* %c,
                                %opencl.image2d_t addrspace(1)* %d,
                                %opencl.image2d_array_t addrspace(1)* %e,
                                %opencl.image2d_array_depth_t addrspace(1)* %f,
                                %opencl.image2d_array_msaa_t addrspace(1)* %g,
                                %opencl.image2d_array_msaa_depth_t addrspace(1)* %h,
                                %opencl.image2d_depth_t addrspace(1)* %i,
                                %opencl.image2d_msaa_t addrspace(1)* %j,
                                %opencl.image2d_msaa_depth_t addrspace(1)* %k,
                                %opencl.image3d_t addrspace(1)* %l)
    !kernel_arg_type !1 !kernel_arg_base_type !1 {
  ret void
}

; CHECK:  amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0

!1 = !{!"image1d_t", !"image1d_array_t", !"image1d_buffer_t",
       !"image2d_t", !"image2d_array_t", !"image2d_array_depth_t",
       !"image2d_array_msaa_t", !"image2d_array_msaa_depth_t",
       !"image2d_depth_t", !"image2d_msaa_t", !"image2d_msaa_depth_t",
       !"image3d_t"}
