; RUN: llc < %s -mtriple=r600 -mcpu=redwood | FileCheck %s

; This tests a bug where LegalizeDAG was not checking the target's
; BooleanContents value and always using one for true, when expanding
; setcc to select_cc.
;
; This bug caused the icmp IR instruction to be expanded to two machine
; instructions, when only one is needed.
;

; CHECK: {{^}}setcc_expand:
; CHECK: SET
; CHECK-NOT: CND
define amdgpu_kernel void @setcc_expand(ptr addrspace(1) %out, i32 %in) {
entry:
  %0 = icmp eq i32 %in, 5
  br i1 %0, label %IF, label %ENDIF
IF:
  %1 = getelementptr i32, ptr addrspace(1) %out, i32 1
  store i32 0, ptr addrspace(1) %1
  br label %ENDIF

ENDIF:
  store i32 0, ptr addrspace(1) %out
  ret void
}
