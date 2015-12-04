; RUN: llc -march=r600 -mcpu=juniper < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; === WIDTH ==================================================================
; 9 implicit args = 9 dwords to first image argument.
; First width at dword index 9+1 -> KC0[2].Z

; FUNC-LABEL: {{^}}width_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].Z
define void @width_2d (%opencl.image2d_t addrspace(1)* %in,
                       i32 addrspace(1)* %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.2d(
      %opencl.image2d_t addrspace(1)* %in) #0
  %1 = extractvalue [3 x i32] %0, 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}width_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].Z
define void @width_3d (%opencl.image3d_t addrspace(1)* %in,
                       i32 addrspace(1)* %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.3d(
      %opencl.image3d_t addrspace(1)* %in) #0
  %1 = extractvalue [3 x i32] %0, 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}


; === HEIGHT =================================================================
; First height at dword index 9+2 -> KC0[2].W

; FUNC-LABEL: {{^}}height_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].W
define void @height_2d (%opencl.image2d_t addrspace(1)* %in,
                        i32 addrspace(1)* %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.2d(
      %opencl.image2d_t addrspace(1)* %in) #0
  %1 = extractvalue [3 x i32] %0, 1
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}height_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[2].W
define void @height_3d (%opencl.image3d_t addrspace(1)* %in,
                        i32 addrspace(1)* %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.3d(
      %opencl.image3d_t addrspace(1)* %in) #0
  %1 = extractvalue [3 x i32] %0, 1
  store i32 %1, i32 addrspace(1)* %out
  ret void
}


; === DEPTH ==================================================================
; First depth at dword index 9+3 -> KC0[3].X

; FUNC-LABEL: {{^}}depth_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].X
define void @depth_3d (%opencl.image3d_t addrspace(1)* %in,
                       i32 addrspace(1)* %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.3d(
      %opencl.image3d_t addrspace(1)* %in) #0
  %1 = extractvalue [3 x i32] %0, 2
  store i32 %1, i32 addrspace(1)* %out
  ret void
}


; === CHANNEL DATA TYPE ======================================================
; First channel data type at dword index 9+4 -> KC0[3].Y

; FUNC-LABEL: {{^}}data_type_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Y
define void @data_type_2d (%opencl.image2d_t addrspace(1)* %in,
                           i32 addrspace(1)* %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.2d(
      %opencl.image2d_t addrspace(1)* %in) #0
  %1 = extractvalue [2 x i32] %0, 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}data_type_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Y
define void @data_type_3d (%opencl.image3d_t addrspace(1)* %in,
                                     i32 addrspace(1)* %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.3d(
      %opencl.image3d_t addrspace(1)* %in) #0
  %1 = extractvalue [2 x i32] %0, 0
  store i32 %1, i32 addrspace(1)* %out
  ret void
}


; === CHANNEL ORDER ==========================================================
; First channel order at dword index 9+5 -> KC0[3].Z

; FUNC-LABEL: {{^}}channel_order_2d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Z
define void @channel_order_2d (%opencl.image2d_t addrspace(1)* %in,
                               i32 addrspace(1)* %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.2d(
      %opencl.image2d_t addrspace(1)* %in) #0
  %1 = extractvalue [2 x i32] %0, 1
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}channel_order_3d:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV * [[VAL]], KC0[3].Z
define void @channel_order_3d (%opencl.image3d_t addrspace(1)* %in,
                                         i32 addrspace(1)* %out) {
entry:
  %0 = call [2 x i32] @llvm.OpenCL.image.get.format.3d(
      %opencl.image3d_t addrspace(1)* %in) #0
  %1 = extractvalue [2 x i32] %0, 1
  store i32 %1, i32 addrspace(1)* %out
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
define void @image_arg_2nd (%opencl.image3d_t addrspace(1)* %in1,
                            i32 %x,
                            %opencl.image2d_t addrspace(1)* %in2,
                            i32 addrspace(1)* %out) {
entry:
  %0 = call [3 x i32] @llvm.OpenCL.image.get.size.2d(
      %opencl.image2d_t addrspace(1)* %in2) #0
  %1 = extractvalue [3 x i32] %0, 1
  store i32 %1, i32 addrspace(1)* %out
  ret void
}

%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque

declare [3 x i32] @llvm.OpenCL.image.get.size.2d(%opencl.image2d_t addrspace(1)*) #0
declare [3 x i32] @llvm.OpenCL.image.get.size.3d(%opencl.image3d_t addrspace(1)*) #0
declare [2 x i32] @llvm.OpenCL.image.get.format.2d(%opencl.image2d_t addrspace(1)*) #0
declare [2 x i32] @llvm.OpenCL.image.get.format.3d(%opencl.image3d_t addrspace(1)*) #0

attributes #0 = { readnone }

!opencl.kernels = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!0 = !{void (%opencl.image2d_t addrspace(1)*, i32 addrspace(1)*)* @width_2d,
       !10, !20, !30, !40, !50}
!1 = !{void (%opencl.image3d_t addrspace(1)*, i32 addrspace(1)*)* @width_3d,
       !10, !21, !31, !41, !50}
!2 = !{void (%opencl.image2d_t addrspace(1)*, i32 addrspace(1)*)* @height_2d,
       !10, !20, !30, !40, !50}
!3 = !{void (%opencl.image3d_t addrspace(1)*, i32 addrspace(1)*)* @height_3d,
       !10, !21, !31, !41, !50}
!4 = !{void (%opencl.image3d_t addrspace(1)*, i32 addrspace(1)*)* @depth_3d,
       !10, !21, !31, !41, !50}
!5 = !{void (%opencl.image2d_t addrspace(1)*, i32 addrspace(1)*)* @data_type_2d,
       !10, !20, !30, !40, !50}
!6 = !{void (%opencl.image3d_t addrspace(1)*, i32 addrspace(1)*)* @data_type_3d,
       !10, !21, !31, !41, !50}
!7 = !{void (%opencl.image2d_t addrspace(1)*, i32 addrspace(1)*)* @channel_order_2d,
       !10, !20, !30, !40, !50}
!8 = !{void (%opencl.image3d_t addrspace(1)*, i32 addrspace(1)*)* @channel_order_3d,
       !10, !21, !31, !41, !50}
!9 = !{void (%opencl.image3d_t addrspace(1)*, i32, %opencl.image2d_t addrspace(1)*,
      i32 addrspace(1)*)* @image_arg_2nd, !12, !22, !32, !42, !52}

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
