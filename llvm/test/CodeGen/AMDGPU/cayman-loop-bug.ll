; RUN: llc < %s -march=r600 -mcpu=cayman | FileCheck %s

; CHECK-LABEL: {{^}}main:
; CHECK: LOOP_START_DX10
; CHECK: ALU_PUSH_BEFORE
; CHECK: LOOP_START_DX10
; CHECK: PUSH
; CHECK-NOT: ALU_PUSH_BEFORE
; CHECK: END_LOOP
; CHECK: END_LOOP
define amdgpu_ps void @main (<4 x float> inreg %reg0) {
entry:
  br label %outer_loop

outer_loop:
  %cnt = phi i32 [0, %entry], [%cnt_incr, %inner_loop]
  %cond = icmp eq i32 %cnt, 16
  br i1 %cond, label %outer_loop_body, label %exit

outer_loop_body:
  %cnt_incr = add i32 %cnt, 1
  br label %inner_loop

inner_loop:
  %cnt2 = phi i32 [0, %outer_loop_body], [%cnt2_incr, %inner_loop_body]
  %n = load volatile i32, ptr addrspace(1) undef
  %cond2 = icmp slt i32 %cnt2, %n
  br i1 %cond2, label %inner_loop_body, label %outer_loop

inner_loop_body:
  %cnt2_incr = add i32 %cnt2, 1
  br label %inner_loop

exit:
  ret void
}
