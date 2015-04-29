; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=FUNC %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck --check-prefix=SI --check-prefix=FUNC %s

; FUNC-LABEL: {{^}}break_inserted_outside_of_loop:

; SI: [[LOOP_LABEL:[A-Z0-9]+]]:
; Lowered break instructin:
; SI: s_or_b64
; Lowered Loop instruction:
; SI: s_andn2_b64
; s_cbranch_execnz [[LOOP_LABEL]]
; SI: s_endpgm
define void @break_inserted_outside_of_loop(i32 addrspace(1)* %out, i32 %a, i32 %b) {
main_body:
  %0 = and i32 %a, %b
  %1 = trunc i32 %0 to i1
  br label %ENDIF

ENDLOOP:
  store i32 0, i32 addrspace(1)* %out
  ret void

ENDIF:
  br i1 %1, label %ENDLOOP, label %ENDIF
}
