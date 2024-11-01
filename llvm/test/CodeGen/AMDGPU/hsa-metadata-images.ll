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

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"amdgpu_code_object_version", i32 200}
!1 = !{!"image1d_t", !"image1d_array_t", !"image1d_buffer_t",
       !"image2d_t", !"image2d_array_t", !"image2d_array_depth_t",
       !"image2d_array_msaa_t", !"image2d_array_msaa_depth_t",
       !"image2d_depth_t", !"image2d_msaa_t", !"image2d_msaa_depth_t",
       !"image3d_t"}
