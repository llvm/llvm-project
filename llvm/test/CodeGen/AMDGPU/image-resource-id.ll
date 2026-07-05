; RUN: llc -mtriple=r600 -mcpu=juniper < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; === 1 image arg, read_only ===================================================

; FUNC-LABEL: {{^}}test_2d_rd_1_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_2d_rd_1_0(ptr addrspace(1) %in, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_rd_1_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_3d_rd_1_0(ptr addrspace(1) %in, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; === 1 image arg, write_only ==================================================

; FUNC-LABEL: {{^}}test_2d_wr_1_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_2d_wr_1_0(ptr addrspace(1) %in, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_wr_1_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_3d_wr_1_0(ptr addrspace(1) %in, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; === 2 image args, read_only ==================================================

; FUNC-LABEL: {{^}}test_2d_rd_2_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_2d_rd_2_0(ptr addrspace(1) %in1, ; read_only
                            ptr addrspace(1) %in2, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in1) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_2d_rd_2_1:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_2d_rd_2_1(ptr addrspace(1) %in1, ; read_only
                            ptr addrspace(1) %in2, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in2) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_rd_2_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_3d_rd_2_0(ptr addrspace(1) %in1, ; read_only
                            ptr addrspace(1) %in2, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in1) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_rd_2_1:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_3d_rd_2_1(ptr addrspace(1) %in1, ; read_only
                            ptr addrspace(1) %in2, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in2) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; === 2 image args, write_only =================================================

; FUNC-LABEL: {{^}}test_2d_wr_2_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_2d_wr_2_0(ptr addrspace(1) %in1, ; write_only
                            ptr addrspace(1) %in2, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in1) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_2d_wr_2_1:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_2d_wr_2_1(ptr addrspace(1) %in1, ; write_only
                            ptr addrspace(1) %in2, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in2) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_wr_2_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_3d_wr_2_0(ptr addrspace(1) %in1, ; write_only
                            ptr addrspace(1) %in2, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in1) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_wr_2_1:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_3d_wr_2_1(ptr addrspace(1) %in1, ; write_only
                            ptr addrspace(1) %in2, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in2) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; === 3 image args, read_only ==================================================

; FUNC-LABEL: {{^}}test_2d_rd_3_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 2(
define amdgpu_kernel void @test_2d_rd_3_0(ptr addrspace(1) %in1, ; read_only
                            ptr addrspace(1) %in2, ; read_only
                            ptr addrspace(1) %in3, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}


; FUNC-LABEL: {{^}}test_3d_rd_3_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 2(
define amdgpu_kernel void @test_3d_rd_3_0(ptr addrspace(1) %in1, ; read_only
                            ptr addrspace(1) %in2, ; read_only
                            ptr addrspace(1) %in3, ; read_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; === 3 image args, write_only =================================================

; FUNC-LABEL: {{^}}test_2d_wr_3_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 2(
define amdgpu_kernel void @test_2d_wr_3_0(ptr addrspace(1) %in1, ; write_only
                            ptr addrspace(1) %in2, ; write_only
                            ptr addrspace(1) %in3, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}


; FUNC-LABEL: {{^}}test_3d_wr_3_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 2(
define amdgpu_kernel void @test_3d_wr_3_0(ptr addrspace(1) %in1, ; write_only
                            ptr addrspace(1) %in2, ; write_only
                            ptr addrspace(1) %in3, ; write_only
                            ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; === 3 image args, mixed ======================================================

; FUNC-LABEL: {{^}}test_2d_mix_3_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_2d_mix_3_0(ptr addrspace(1) %in1, ; write_only
                             ptr addrspace(1) %in2, ; read_only
                             ptr addrspace(1) %in3, ; read_only
                             ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_mix_3_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_3d_mix_3_0(ptr addrspace(1) %in1, ; write_only
                             ptr addrspace(1) %in2, ; read_only
                             ptr addrspace(1) %in3, ; read_only
                             ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_2d_mix_3_1:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_2d_mix_3_1(ptr addrspace(1) %in1, ; write_only
                             ptr addrspace(1) %in2, ; read_only
                             ptr addrspace(1) %in3, ; write_only
                             ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.2d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_3d_mix_3_1:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_3d_mix_3_1(ptr addrspace(1) %in1, ; write_only
                             ptr addrspace(1) %in2, ; read_only
                             ptr addrspace(1) %in3, ; write_only
                             ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.image.get.resource.id.3d(
      ptr addrspace(1) %in3) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}


%opencl.image2d_t = type opaque
%opencl.image3d_t = type opaque

declare i32 @llvm.OpenCL.image.get.resource.id.2d(ptr addrspace(1)) #0
declare i32 @llvm.OpenCL.image.get.resource.id.3d(ptr addrspace(1)) #0

attributes #0 = { readnone }

!opencl.kernels = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13,
                    !14, !15, !16, !17, !18, !19}
!0 = !{ptr @test_2d_rd_1_0,
       !110, !120, !130, !140, !150}
!1 = !{ptr @test_3d_rd_1_0,
       !110, !120, !131, !141, !150}
!2 = !{ptr @test_2d_wr_1_0,
       !110, !121, !130, !140, !150}
!3 = !{ptr @test_3d_wr_1_0,
       !110, !121, !131, !141, !150}
!110 = !{!"kernel_arg_addr_space", i32 1, i32 1}
!120 = !{!"kernel_arg_access_qual", !"read_only", !"none"}
!121 = !{!"kernel_arg_access_qual", !"write_only", !"none"}
!130 = !{!"kernel_arg_type", !"image2d_t", !"int*"}
!131 = !{!"kernel_arg_type", !"image3d_t", !"int*"}
!140 = !{!"kernel_arg_base_type", !"image2d_t", !"int*"}
!141 = !{!"kernel_arg_base_type", !"image3d_t", !"int*"}
!150 = !{!"kernel_arg_type_qual", !"", !""}

!4  = !{ptr @test_2d_rd_2_0, !112, !122, !132, !142, !152}
!5  = !{ptr @test_2d_rd_2_1, !112, !122, !132, !142, !152}
!6  = !{ptr @test_3d_rd_2_0, !112, !122, !133, !143, !152}
!7  = !{ptr @test_3d_rd_2_1, !112, !122, !133, !143, !152}
!8  = !{ptr @test_2d_wr_2_0, !112, !123, !132, !142, !152}
!9  = !{ptr @test_2d_wr_2_1, !112, !123, !132, !142, !152}
!10 = !{ptr @test_3d_wr_2_0, !112, !123, !133, !143, !152}
!11 = !{ptr @test_3d_wr_2_1, !112, !123, !133, !143, !152}
!112 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1}
!122 = !{!"kernel_arg_access_qual", !"read_only", !"read_only", !"none"}
!123 = !{!"kernel_arg_access_qual", !"write_only", !"write_only", !"none"}
!132 = !{!"kernel_arg_type", !"image2d_t", !"image2d_t", !"int*"}
!133 = !{!"kernel_arg_type", !"image3d_t", !"image3d_t", !"int*"}
!142 = !{!"kernel_arg_base_type", !"image2d_t", !"image2d_t", !"int*"}
!143 = !{!"kernel_arg_base_type", !"image3d_t", !"image3d_t", !"int*"}
!152 = !{!"kernel_arg_type_qual", !"", !"", !""}

!12 = !{ptr @test_2d_rd_3_0,
              !114, !124, !134, !144, !154}
!13 = !{ptr @test_3d_rd_3_0,
              !114, !124, !135, !145, !154}
!14 = !{ptr @test_2d_wr_3_0,
              !114, !125, !134, !144, !154}
!15 = !{ptr @test_3d_wr_3_0,
              !114, !125, !135, !145, !154}
!16 = !{ptr @test_2d_mix_3_0,
              !114, !126, !134, !144, !154}
!17 = !{ptr @test_3d_mix_3_0,
              !114, !126, !135, !145, !154}
!18 = !{ptr @test_2d_mix_3_1,
              !114, !127, !134, !144, !154}
!19 = !{ptr @test_3d_mix_3_1,
              !114, !127, !135, !145, !154}
!114 = !{!"kernel_arg_addr_space", i32 1, i32 1, i32 1, i32 1}
!124 = !{!"kernel_arg_access_qual", !"read_only", !"read_only", !"read_only", !"none"}
!125 = !{!"kernel_arg_access_qual", !"write_only", !"write_only", !"write_only", !"none"}
!126 = !{!"kernel_arg_access_qual", !"write_only", !"read_only", !"read_only", !"none"}
!127 = !{!"kernel_arg_access_qual", !"write_only", !"read_only", !"write_only", !"none"}
!134 = !{!"kernel_arg_type", !"image2d_t", !"image3d_t", !"image2d_t", !"int*"}
!135 = !{!"kernel_arg_type", !"image3d_t", !"image2d_t", !"image3d_t", !"int*"}
!144 = !{!"kernel_arg_base_type", !"image2d_t", !"image3d_t", !"image2d_t", !"int*"}
!145 = !{!"kernel_arg_base_type", !"image3d_t", !"image2d_t", !"image3d_t", !"int*"}
!154 = !{!"kernel_arg_type_qual", !"", !"", !"", !""}
