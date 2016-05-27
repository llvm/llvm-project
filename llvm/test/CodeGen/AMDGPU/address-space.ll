; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck %s

; Test that codegenprepare understands address space sizes

%struct.foo = type { [3 x float], [3 x float] }

; CHECK-LABEL: {{^}}do_as_ptr_calcs:
; CHECK: s_load_dword [[SREG1:s[0-9]+]],
; CHECK: v_mov_b32_e32 [[VREG1:v[0-9]+]], [[SREG1]]
; CHECK-DAG: ds_read2_b32 v[{{[0-9+:[0-9]+}}], [[VREG1]] offset0:3 offset1:5
define void @do_as_ptr_calcs(%struct.foo addrspace(3)* nocapture %ptr) nounwind {
entry:
  %x = getelementptr inbounds %struct.foo, %struct.foo addrspace(3)* %ptr, i32 0, i32 1, i32 0
  %y = getelementptr inbounds %struct.foo, %struct.foo addrspace(3)* %ptr, i32 0, i32 1, i32 2
  br label %bb32

bb32:
  %a = load float, float addrspace(3)* %x, align 4
  %b = load float, float addrspace(3)* %y, align 4
  %cmp = fcmp one float %a, %b
  br i1 %cmp, label %bb34, label %bb33

bb33:
  unreachable

bb34:
  unreachable
}


