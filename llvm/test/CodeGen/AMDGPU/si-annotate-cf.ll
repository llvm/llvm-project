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

; FIXME: should emit s_endpgm
; CHECK-LABEL: {{^}}switch_unreachable:
; CHECK-NOT: s_endpgm
; CHECK: .Lfunc_end2
define void @switch_unreachable(i32 addrspace(1)* %g, i8 addrspace(3)* %l, i32 %x) nounwind {
centry:
  switch i32 %x, label %sw.default [
    i32 0, label %sw.bb
    i32 60, label %sw.bb
  ]

sw.bb:
  unreachable

sw.default:
  unreachable

sw.epilog:
  ret void
}

declare float @llvm.fabs.f32(float) nounwind readnone

; This broke the old AMDIL cfg structurizer
; FUNC-LABEL: {{^}}loop_land_info_assert:
; SI: s_cmp_gt_i32
; SI-NEXT: s_cbranch_scc0 [[ENDPGM:BB[0-9]+_[0-9]+]]

; SI: s_cmp_gt_i32
; SI-NEXT: s_cbranch_scc1 [[ENDPGM]]

; SI: [[INFLOOP:BB[0-9]+_[0-9]+]]
; SI: s_branch [[INFLOOP]]

; SI: [[ENDPGM]]:
; SI: s_endpgm
define void @loop_land_info_assert(i32 %c0, i32 %c1, i32 %c2, i32 %c3, i32 %x, i32 %y, i1 %arg) nounwind {
entry:
  %cmp = icmp sgt i32 %c0, 0
  br label %while.cond.outer

while.cond.outer:
  %tmp = load float, float addrspace(1)* undef
  br label %while.cond

while.cond:
  %cmp1 = icmp slt i32 %c1, 4
  br i1 %cmp1, label %convex.exit, label %for.cond

convex.exit:
  %or = or i1 %cmp, %cmp1
  br i1 %or, label %return, label %if.end

if.end:
  %tmp3 = call float @llvm.fabs.f32(float %tmp) nounwind readnone
  %cmp2 = fcmp olt float %tmp3, 0x3E80000000000000
  br i1 %cmp2, label %if.else, label %while.cond.outer

if.else:
  store volatile i32 3, i32 addrspace(1)* undef, align 4
  br label %while.cond

for.cond:
  %cmp3 = icmp slt i32 %c3, 1000
  br i1 %cmp3, label %for.body, label %return

for.body:
  br i1 %cmp3, label %self.loop, label %if.end.2

if.end.2:
  %or.cond2 = or i1 %cmp3, %arg
  br i1 %or.cond2, label %return, label %for.cond

self.loop:
 br label %self.loop

return:
  ret void
}


declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #0

attributes #0 = { nounwind readnone }
