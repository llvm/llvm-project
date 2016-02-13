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
define void @break_inserted_outside_of_loop(i32 addrspace(1)* %out, i32 %a) {
main_body:
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %0 = and i32 %a, %tid
  %1 = trunc i32 %0 to i1
  br label %ENDIF

ENDLOOP:
  store i32 0, i32 addrspace(1)* %out
  ret void

ENDIF:
  br i1 %1, label %ENDLOOP, label %ENDIF
}


; FUNC-LABEL: {{^}}phi_cond_outside_loop:
; FIXME: This could be folded into the s_or_b64 instruction
; SI: s_mov_b64 [[ZERO:s\[[0-9]+:[0-9]+\]]], 0
; SI: [[LOOP_LABEL:[A-Z0-9]+]]
; SI: v_cmp_ne_i32_e32 vcc, 0, v{{[0-9]+}}

; SI_IF_BREAK instruction:
; SI: s_or_b64 [[BREAK:s\[[0-9]+:[0-9]+\]]], vcc, [[ZERO]]

; SI_LOOP instruction:
; SI: s_andn2_b64 exec, exec, [[BREAK]]
; SI: s_cbranch_execnz [[LOOP_LABEL]]
; SI: s_endpgm

define void @phi_cond_outside_loop(i32 %b) {
entry:
  %tid = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0) #0
  %0 = icmp eq i32 %tid , 0
  br i1 %0, label %if, label %else

if:
  br label %endif

else:
  %1 = icmp eq i32 %b, 0
  br label %endif

endif:
  %2 = phi i1 [0, %if], [%1, %else]
  br label %loop

loop:
  br i1 %2, label %exit, label %loop

exit:
  ret void
}

declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0

attributes #0 = { nounwind readnone }
