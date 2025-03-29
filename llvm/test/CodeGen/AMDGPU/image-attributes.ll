; RUN: llc -mtriple=r600 -mcpu=juniper < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; === WIDTH ==================================================================
; 9 implicit args = 9 dwords to first image argument.
; First width at dword index 9+1 -> KC0[2].Z

; FUNC-LABEL: {{^}}width_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].Z
define amdgpu_kernel void @width_2d (ptr addrspace(1) %in,
                       ptr addrspace(1) %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.2d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [3 x i32] %0, 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}width_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].Z
define amdgpu_kernel void @width_3d (ptr addrspace(1) %in,
                       ptr addrspace(1) %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.3d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [3 x i32] %0, 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}


; === HEIGHT =================================================================
; First height at dword index 9+2 -> KC0[2].W

; FUNC-LABEL: {{^}}height_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].W
define amdgpu_kernel void @height_2d (ptr addrspace(1) %in,
                        ptr addrspace(1) %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.2d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [3 x i32] %0, 1
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}height_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].W
define amdgpu_kernel void @height_3d (ptr addrspace(1) %in,
                        ptr addrspace(1) %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.3d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [3 x i32] %0, 1
  store i32 %1, ptr addrspace(1) %out
  ret void
}


; === DEPTH ==================================================================
; First depth at dword index 9+3 -> KC0[3].X

; FUNC-LABEL: {{^}}depth_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].X
define amdgpu_kernel void @depth_3d (ptr addrspace(1) %in,
                       ptr addrspace(1) %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.3d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [3 x i32] %0, 2
  store i32 %1, ptr addrspace(1) %out
  ret void
}


; === CHANNEL DATA TYPE ======================================================
; First channel data type at dword index 9+4 -> KC0[3].Y

; FUNC-LABEL: {{^}}data_type_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Y
define amdgpu_kernel void @data_type_2d (ptr addrspace(1) %in,
                           ptr addrspace(1) %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.2d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [2 x i32] %0, 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}data_type_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Y
define amdgpu_kernel void @data_type_3d (ptr addrspace(1) %in,
                                     ptr addrspace(1) %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.3d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [2 x i32] %0, 0
  store i32 %1, ptr addrspace(1) %out
  ret void
}


; === CHANNEL ORDER ==========================================================
; First channel order at dword index 9+5 -> KC0[3].Z

; FUNC-LABEL: {{^}}channel_order_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Z
define amdgpu_kernel void @channel_order_2d (ptr addrspace(1) %in,
                               ptr addrspace(1) %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.2d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [2 x i32] %0, 1
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}channel_order_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Z
define amdgpu_kernel void @channel_order_3d (ptr addrspace(1) %in,
                                         ptr addrspace(1) %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.3d(
      ptr addrspace(1) %in) #0
  %1 = extractvalue [2 x i32] %0, 1
  store i32 %1, ptr addrspace(1) %out
  ret void
}


; === 2ND IMAGE ==============================================================
; 9 implicit args + 2 explicit args + 5 implicit args for 1st image argument
;   = 16 dwords to 2nd image argument.
; Height of the second image is at 16+2 -> KC0[4].Z
;
; FUNC-LABEL: {{^}}image_arg_2nd:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[4].Z
define amdgpu_kernel void @image_arg_2nd (ptr addrspace(1) %in1,
                            i32 %x,
                            ptr addrspace(1) %in2,
                            ptr addrspace(1) %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.2d(
      ptr addrspace(1) %in2) #0
  %1 = extractvalue [3 x i32] %0, 1
  store i32 %1, ptr addrspace(1) %out
  ret void
}

%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque

declare [3 x i32] @llvm.OpenCL.image.get.size.2d(ptr addrspace(1)) #0
declare [3 x i32] @llvm.OpenCL.image.get.size.3d(ptr addrspace(1)) #0
declare [2 x i32] @llvm.OpenCL.image.get.format.2d(ptr addrspace(1)) #0
declare [2 x i32] @llvm.OpenCL.image.get.format.3d(ptr addrspace(1)) #0

attributes #0 = { readnone }

!opencl.kernels = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!0 = !{ptr @width_2d,
       !10, !20, !30, !40, !50}
!1 = !{ptr @width_3d,
       !10, !21, !31, !41, !50}
!2 = !{ptr @height_2d,
       !10, !20, !30, !40, !50}
!3 = !{ptr @height_3d,
       !10, !21, !31, !41, !50}
!4 = !{ptr @depth_3d,
       !10, !21, !31, !41, !50}
!5 = !{ptr @data_type_2d,
       !10, !20, !30, !40, !50}
!6 = !{ptr @data_type_3d,
       !10, !21, !31, !41, !50}
!7 = !{ptr @channel_order_2d,
       !10, !20, !30, !40, !50}
!8 = !{ptr @channel_order_3d,
       !10, !21, !31, !41, !50}
!9 = !{ptr @image_arg_2nd, !12, !22, !32, !42, !52}

!10 = !{!"kernel_arg_addr_space", i32 1, i32 1}
!20 = !{!"kernel_arg_access_qual", !"read_only", !"none"}
!21 = !{!"kernel_arg_access_qual", !"read_only", !"none"}
!30 = !{!"kernel_arg_type", !"image2d_t", !"int*"}
!31 = !{!"kernel_arg_type", !"image3d_t", !"int*"}
!40 = !{!"kernel_arg_base_type", !"image2d_t", !"int*"}
!41 = !{!"kernel_arg_base_type", !"image3d_t", !"int*"}
!50 = !{!"kernel_arg_type_qual", !"", !""}

!12 = !{!"kernel_arg_addr_space", i32 1, i32 0, i32 1, i32 1}
!22 = !{!"kernel_arg_access_qual", !"read_only", !"none", !"write_only", !"none"}
!32 = !{!"kernel_arg_type", !"image3d_t", !"sampler_t", !"image2d_t", !"int*"}
!42 = !{!"kernel_arg_base_type", !"image3d_t", !"sampler_t", !"image2d_t", !"int*"}
!52 = !{!"kernel_arg_type_qual", !"", !"", !"", !""}
