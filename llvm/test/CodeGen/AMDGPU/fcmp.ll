; RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

; CHECK: {{^}}fcmp_sext:
; CHECK: SETE_DX10  T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define amdgpu_kernel void @fcmp_sext(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %0 = load float, ptr addrspace(1) %in
  %arrayidx1 = getelementptr inbounds float, ptr addrspace(1) %in, i32 1
  %1 = load float, ptr addrspace(1) %arrayidx1
  %cmp = fcmp oeq float %0, %1
  %sext = sext i1 %cmp to i32
  store i32 %sext, ptr addrspace(1) %out
  ret void
}

; This test checks that a setcc node with f32 operands is lowered to a
; SET*_DX10 instruction.  Previously we were lowering this to:
; SET* + FP_TO_SINT

; CHECK: {{^}}fcmp_br:
; CHECK: SET{{[N]*}}E_DX10 * T{{[0-9]+\.[XYZW],}}
; CHECK-NEXT: {{[0-9]+\(5.0}}

define amdgpu_kernel void @fcmp_br(ptr addrspace(1) %out, float %in) {
entry:
  %0 = fcmp oeq float %in, 5.0
  br i1 %0, label %IF, label %ENDIF

IF:
  %1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 0, ptr addrspace(1) %1
  br label %ENDIF

ENDIF:
  store i32 0, ptr addrspace(1) %out
  ret void
}
