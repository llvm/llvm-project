; RUN: opt -mtriple amdgcn-- -passes='print<uniformity>' -disable-output %s 2>&1 | FileCheck %s

; CHECK-LABEL: for function 'test_amdgpu_ps':
; CHECK-DAG: DIVERGENT:  ptr addrspace(4) %arg0
; CHECK-DAG: DIVERGENT:  <2 x i32> %arg3
; CHECK-DAG: DIVERGENT:  <3 x i32> %arg4
; CHECK-DAG: DIVERGENT:  float %arg5
; CHECK-DAG: DIVERGENT:  i32 %arg6
; CHECK-NOT: DIVERGENT

define amdgpu_ps void @test_amdgpu_ps(ptr addrspace(4) byref([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

; CHECK-LABEL: for function 'test_amdgpu_kernel':
; CHECK-NOT: %arg0
; CHECK-NOT: %arg1
; CHECK-NOT: %arg2
; CHECK-NOT: %arg3
; CHECK-NOT: %arg4
; CHECK-NOT: %arg5
; CHECK-NOT: %arg6
define amdgpu_kernel void @test_amdgpu_kernel(ptr addrspace(4) byref([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

; CHECK-LABEL: for function 'test_c':
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
; CHECK: DIVERGENT:
define void @test_c(ptr addrspace(5) byval([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

; CHECK-LABEL: for function 'test_amdgpu_cs_chain':
; CHECK-DAG: DIVERGENT:  ptr addrspace(4) %arg0
; CHECK-DAG: DIVERGENT:  <2 x i32> %arg3
; CHECK-DAG: DIVERGENT:  <3 x i32> %arg4
; CHECK-DAG: DIVERGENT:  float %arg5
; CHECK-DAG: DIVERGENT:  i32 %arg6
; CHECK-NOT: DIVERGENT
define amdgpu_cs_chain void @test_amdgpu_cs_chain(ptr addrspace(4) byref([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}

; CHECK-LABEL: for function 'test_amdgpu_cs_chain_preserve':
; CHECK-DAG: DIVERGENT:  ptr addrspace(4) %arg0
; CHECK-DAG: DIVERGENT:  <2 x i32> %arg3
; CHECK-DAG: DIVERGENT:  <3 x i32> %arg4
; CHECK-DAG: DIVERGENT:  float %arg5
; CHECK-DAG: DIVERGENT:  i32 %arg6
; CHECK-NOT: DIVERGENT
define amdgpu_cs_chain_preserve void @test_amdgpu_cs_chain_preserve(ptr addrspace(4) byref([4 x <16 x i8>]) %arg0, float inreg %arg1, i32 inreg %arg2, <2 x i32> %arg3, <3 x i32> %arg4, float %arg5, i32 %arg6) #0 {
  ret void
}
attributes #0 = { nounwind }
