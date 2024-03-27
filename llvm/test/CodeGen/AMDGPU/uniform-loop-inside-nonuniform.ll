; XFAIL: *
; RUN: llc -mtriple=amdgcn -mcpu=verde < %s | FileCheck %s

; Test a simple uniform loop that lives inside non-uniform control flow.

; CHECK-LABEL: {{^}}test1:
; CHECK: v_cmp_ne_u32_e32 vcc, 0
; CHECK: s_and_saveexec_b64
; CHECK-NEXT: s_cbranch_execz .LBB{{[0-9]+_[0-9]+}}

; CHECK: [[LOOP_BODY_LABEL:.LBB[0-9]+_[0-9]+]]: ; %loop_body
; CHECK: s_cbranch_scc1 [[LOOP_BODY_LABEL]]

; CHECK: s_endpgm
define amdgpu_ps void @test1(<8 x i32> inreg %rsrc, <2 x i32> %addr.base, i32 %y, i32 %p) {
main_body:
  %cc = icmp eq i32 %p, 0
  br i1 %cc, label %out, label %loop_body

loop_body:
  %counter = phi i32 [ 0, %main_body ], [ %incr, %loop_body ]

  ; Prevent the loop from being optimized out
  call void asm sideeffect "", "" ()

  %incr = add i32 %counter, 1
  %lc = icmp sge i32 %incr, 1000
  br i1 %lc, label %out, label %loop_body

out:
  ret void
}

; CHECK-LABEL: {{^}}test2:
; CHECK: s_and_saveexec_b64
; CHECK-NEXT: s_cbranch_execz
define amdgpu_kernel void @test2(ptr addrspace(1) %out, i32 %a, i32 %b) {
main_body:
  %tid = call i32 @llvm.amdgcn.workitem.id.x() #1
  %cc = icmp eq i32 %tid, 0
  br i1 %cc, label %done1, label %if

if:
  %cmp = icmp eq i32 %a, 0
  br i1 %cmp, label %done0, label %loop_body

loop_body:
  %counter = phi i32 [ 0, %if ], [0, %done0], [ %incr, %loop_body ]

  ; Prevent the loop from being optimized out
  call void asm sideeffect "", "" ()

  %incr = add i32 %counter, 1
  %lc = icmp sge i32 %incr, 1000
  br i1 %lc, label %done1, label %loop_body

done0:
  %cmp0 = icmp eq i32 %b, 0
  br i1 %cmp0, label %done1, label %loop_body

done1:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #1 = { nounwind readonly }
