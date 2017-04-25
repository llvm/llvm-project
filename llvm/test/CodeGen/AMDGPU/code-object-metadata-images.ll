; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx800 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX800 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

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

; CHECK: ---
; CHECK:  Version: [ 1, 0 ]

; CHECK:  Kernels:
; CHECK:    - Name: test
; CHECK:      Args:
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image1d_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image1d_array_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image1d_buffer_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_array_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_array_depth_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_array_msaa_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_array_msaa_depth_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_depth_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_msaa_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image2d_msaa_depth_t
; CHECK:        - Size:      8
; CHECK:          ValueKind: Image
; CHECK:          TypeName:  image3d_t
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

!1 = !{!"image1d_t", !"image1d_array_t", !"image1d_buffer_t",
       !"image2d_t", !"image2d_array_t", !"image2d_array_depth_t",
       !"image2d_array_msaa_t", !"image2d_array_msaa_depth_t",
       !"image2d_depth_t", !"image2d_msaa_t", !"image2d_msaa_depth_t",
       !"image3d_t"}
