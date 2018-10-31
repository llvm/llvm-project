; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX700 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx802 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX802 --check-prefix=NOTES %s
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -filetype=obj -o - < %s | llvm-readobj -elf-output-style=GNU -notes | FileCheck --check-prefix=CHECK --check-prefix=GFX900 --check-prefix=NOTES %s

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
; CHECK:    - Name:       test
; CHECK:      SymbolName: 'test@kd'
; CHECK:      Args:
; CHECK:        - Name:      a
; CHECK:          TypeName:  image1d_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      b
; CHECK:          TypeName:  image1d_array_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      c
; CHECK:          TypeName:  image1d_buffer_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      d
; CHECK:          TypeName:  image2d_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      e
; CHECK:          TypeName:  image2d_array_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      f
; CHECK:          TypeName:  image2d_array_depth_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      g
; CHECK:          TypeName:  image2d_array_msaa_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      h
; CHECK:          TypeName:  image2d_array_msaa_depth_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      i
; CHECK:          TypeName:  image2d_depth_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      j
; CHECK:          TypeName:  image2d_msaa_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      k
; CHECK:          TypeName:  image2d_msaa_depth_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
; CHECK:        - Name:      l
; CHECK:          TypeName:  image3d_t
; CHECK:          Size:      8
; CHECK:          ValueKind: Image
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
