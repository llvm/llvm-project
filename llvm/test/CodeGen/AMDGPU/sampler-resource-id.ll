; RUN: llc -mtriple=r600 -mcpu=juniper < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}test_0:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 0(
define amdgpu_kernel void @test_0(i32 %in0, ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.sampler.get.resource.id(i32 %in0) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_1:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 1(
define amdgpu_kernel void @test_1(i32 %in0, i32 %in1, ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.sampler.get.resource.id(i32 %in1) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}

; FUNC-LABEL: {{^}}test_2:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], literal.x
; EG-NEXT: LSHR
; EG-NEXT: 2(
define amdgpu_kernel void @test_2(i32 %in0, i32 %in1, i32 %in2, ptr addrspace(1) %out) {
entry:
  %0 = call i32 @llvm.OpenCL.sampler.get.resource.id(i32 %in2) #0
  store i32 %0, ptr addrspace(1) %out
  ret void
}


declare i32 @llvm.OpenCL.sampler.get.resource.id(i32) #0

attributes #0 = { readnone }

!opencl.kernels = !{!0, !1, !2}

!0 = !{ptr @test_0, !10, !20, !30, !40, !50}
!10 = !{!"kernel_arg_addr_space", i32 0, i32 1}
!20 = !{!"kernel_arg_access_qual", !"none", !"none"}
!30 = !{!"kernel_arg_type", !"sampler_t", !"int*"}
!40 = !{!"kernel_arg_base_type", !"sampler_t", !"int*"}
!50 = !{!"kernel_arg_type_qual", !"", !""}

!1 = !{ptr @test_1, !11, !21, !31, !41, !51}
!11 = !{!"kernel_arg_addr_space", i32 0, i32 0, i32 1}
!21 = !{!"kernel_arg_access_qual", !"none", !"none", !"none"}
!31 = !{!"kernel_arg_type", !"sampler_t", !"sampler_t", !"int*"}
!41 = !{!"kernel_arg_base_type", !"sampler_t", !"sampler_t", !"int*"}
!51 = !{!"kernel_arg_type_qual", !"", !"", !""}

!2 = !{ptr @test_2, !12, !22, !32, !42, !52}
!12 = !{!"kernel_arg_addr_space", i32 0, i32 0, i32 0, i32 1}
!22 = !{!"kernel_arg_access_qual", !"none", !"none", !"none", !"none"}
!32 = !{!"kernel_arg_type", !"sampler_t", !"sampler_t", !"sampler_t", !"int*"}
!42 = !{!"kernel_arg_base_type", !"sampler_t", !"sampler_t", !"sampler_t", !"int*"}
!52 = !{!"kernel_arg_type_qual", !"", !"", !"", !""}
