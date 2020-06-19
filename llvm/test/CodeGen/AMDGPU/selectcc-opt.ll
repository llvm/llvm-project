; RUN: llc -march=amdgcn < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s


; FUNC-LABEL: {{^}}test_a:
; EG-NOT: CND
; EG: SET{{[NEQGTL]+}}_DX10

define amdgpu_kernel void @test_a(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp olt float %in, 0.000000e+00
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  %4 = bitcast i32 %3 to float
  %5 = bitcast float %4 to i32
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %IF, label %ENDIF

IF:
  %7 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  store i32 0, i32 addrspace(1)* %7
  br label %ENDIF

ENDIF:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; Same as test_a, but the branch labels are swapped to produce the inverse cc
; for the icmp instruction

; EG-LABEL: {{^}}test_b:
; EG: SET{{[GTEQN]+}}_DX10
; EG-NEXT: PRED_
; EG-NEXT: ALU clause starting
define amdgpu_kernel void @test_b(i32 addrspace(1)* %out, float %in) {
entry:
  %0 = fcmp olt float %in, 0.0
  %1 = select i1 %0, float 1.000000e+00, float 0.000000e+00
  %2 = fsub float -0.000000e+00, %1
  %3 = fptosi float %2 to i32
  %4 = bitcast i32 %3 to float
  %5 = bitcast float %4 to i32
  %6 = icmp ne i32 %5, 0
  br i1 %6, label %ENDIF, label %IF

IF:
  %7 = getelementptr i32, i32 addrspace(1)* %out, i32 1
  store i32 0, i32 addrspace(1)* %7
  br label %ENDIF

ENDIF:
  store i32 0, i32 addrspace(1)* %out
  ret void
}

; Test a CND*_INT instruction with float true/false values
; EG-LABEL: {{^}}test_c:
; EG: CND{{[GTE]+}}_INT
define amdgpu_kernel void @test_c(float addrspace(1)* %out, i32 %in) {
entry:
  %0 = icmp sgt i32 %in, 0
  %1 = select i1 %0, float 2.0, float 3.0
  store float %1, float addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}selectcc_bool:
; SI: s_cmp_lg_u32
; SI: v_cndmask_b32_e64
; SI-NOT: cmp
; SI-NOT: cndmask
define amdgpu_kernel void @selectcc_bool(i32 addrspace(1)* %out, i32 %a, i32 %b) nounwind {
  %icmp0 = icmp ne i32 %a, %b
  %ext = select i1 %icmp0, i32 -1, i32 0
  store i32 %ext, i32 addrspace(1)* %out
  ret void
}
