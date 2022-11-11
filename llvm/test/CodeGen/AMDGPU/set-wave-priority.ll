; RUN: llc -mtriple=amdgcn -amdgpu-set-wave-priority=true -o - %s | \
; RUN:   FileCheck %s

; CHECK-LABEL: no_setprio:
; CHECK-NOT:       s_setprio
; CHECK:           ; return to shader part epilog
define amdgpu_ps <2 x float> @no_setprio(<2 x float> %a, <2 x float> %b) "amdgpu-wave-priority-threshold"="1" {
  %s = fadd <2 x float> %a, %b
  ret <2 x float> %s
}

; CHECK-LABEL: vmem_in_exit_block:
; CHECK:           s_setprio 3
; CHECK:           buffer_load_dwordx2
; CHECK-NEXT:      s_setprio 0
; CHECK:           ; return to shader part epilog
define amdgpu_ps <2 x float> @vmem_in_exit_block(<4 x i32> inreg %p, <2 x float> %x) "amdgpu-wave-priority-threshold"="2" {
  %v = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 0, i32 0)
  %s = fadd <2 x float> %v, %x
  ret <2 x float> %s
}

; CHECK-LABEL: branch:
; CHECK:           s_setprio 3
; CHECK:           s_cbranch_scc0 [[A:.*]]
; CHECK:       {{.*}}:  ; %b
; CHECK:           buffer_load_dwordx2
; CHECK-NEXT:      s_setprio 0
; CHECK:           s_branch [[EXIT:.*]]
; CHECK:       [[A]]:  ; %a
; CHECK-NEXT:      s_setprio 0
; CHECK:           s_branch [[EXIT]]
; CHECK-NEXT:  [[EXIT]]:
define amdgpu_ps <2 x float> @branch(<4 x i32> inreg %p, i32 inreg %i, <2 x float> %x) "amdgpu-wave-priority-threshold"="2" {
  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %a, label %b

a:
  ret <2 x float> <float 0.0, float 0.0>

b:
  %v = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 0, i32 0)
  %s = fadd <2 x float> %v, %x
  ret <2 x float> %s
}

; CHECK-LABEL: setprio_follows_setprio:
; CHECK:           s_setprio 3
; CHECK:           buffer_load_dwordx2
; CHECK:           s_cbranch_scc1 [[C:.*]]
; CHECK:       {{.*}}:  ; %a
; CHECK:           buffer_load_dwordx2
; CHECK-NEXT:      s_setprio 0
; CHECK:           s_cbranch_vccnz [[C]]
; CHECK:       {{.*}}:  ; %b
; CHECK-NOT:       s_setprio
; CHECK:           s_branch [[EXIT:.*]]
; CHECK:       [[C]]:  ; %c
; CHECK-NEXT:      s_setprio 0
; CHECK:           s_branch [[EXIT]]
; CHECK:       [[EXIT]]:
define amdgpu_ps <2 x float> @setprio_follows_setprio(<4 x i32> inreg %p, i32 inreg %i) "amdgpu-wave-priority-threshold"="3" {
entry:
  %v1 = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 0, i32 0)
  %cond1 = icmp ne i32 %i, 0
  br i1 %cond1, label %a, label %c

a:
  %v2 = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 1, i32 0)
  %v20 = extractelement <2 x float> %v2, i32 0
  %v21 = extractelement <2 x float> %v2, i32 1
  %cond2 = fcmp ult float %v20, %v21
  br i1 %cond2, label %b, label %c

b:
  ret <2 x float> %v2

c:
  %v4 = fadd <2 x float> %v1, %v1
  ret <2 x float> %v4
}

; CHECK-LABEL: loop:
; CHECK:       {{.*}}:  ; %entry
; CHECK:           s_setprio 3
; CHECK-NOT:       s_setprio
; CHECK:       [[LOOP:.*]]:  ; %loop
; CHECK-NOT:       s_setprio
; CHECK:           buffer_load_dwordx2
; CHECK-NOT:       s_setprio
; CHECK:           s_cbranch_scc1 [[LOOP]]
; CHECK-NEXT:  {{.*}}:  ; %exit
; CHECK-NEXT:      s_setprio 0
define amdgpu_ps <2 x float> @loop(<4 x i32> inreg %p) "amdgpu-wave-priority-threshold"="2" {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%i2, %loop]
  %sum = phi <2 x float> [<float 0.0, float 0.0>, %entry], [%sum2, %loop]

  %i2 = add i32 %i, 1

  %v = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 %i, i32 0, i32 0, i32 0)
  %sum2 = fadd <2 x float> %sum, %v

  %cond = icmp ult i32 %i2, 5
  br i1 %cond, label %loop, label %exit

exit:
  ret <2 x float> %sum2
}

; CHECK-LABEL: edge_split:
; CHECK:           s_setprio 3
; CHECK:           buffer_load_dwordx2
; CHECK-NOT:       s_setprio
; CHECK:           s_cbranch_scc1 [[ANOTHER_LOAD:.*]]
; CHECK:       {{.*}}:  ; %loop.preheader
; CHECK-NEXT:      s_setprio 0
; CHECK:       [[LOOP:.*]]:  ; %loop
; CHECK-NOT:       s_setprio
; CHECK:           s_cbranch_scc1 [[LOOP]]
; CHECK        {{.*}}:  ; %exit
; CHECK-NOT:       s_setprio
; CHECK:           s_branch [[RET:.*]]
; CHECK:       [[ANOTHER_LOAD]]:  ; %another_load
; CHECK:           buffer_load_dwordx2
; CHECK-NEXT:      s_setprio 0
; CHECK:           s_branch [[RET]]
; CHECK:       [[RET]]:
define amdgpu_ps <2 x float> @edge_split(<4 x i32> inreg %p, i32 inreg %x) "amdgpu-wave-priority-threshold"="2" {
entry:
  %v = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 0, i32 0)
  %cond = icmp ne i32 %x, 0
  br i1 %cond, label %loop, label %another_load

loop:
  %i = phi i32 [0, %entry], [%i2, %loop]
  %mul = phi <2 x float> [%v, %entry], [%mul2, %loop]

  %i2 = add i32 %i, 1
  %mul2 = fmul <2 x float> %mul, %v

  %cond2 = icmp ult i32 %i2, 5
  br i1 %cond2, label %loop, label %exit

exit:
  ret <2 x float> %mul2

another_load:
  %v2 = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 1, i32 0)
  %sum = fadd <2 x float> %v, %v2
  ret <2 x float> %sum
}

; CHECK-LABEL: valu_insts_threshold:
; CHECK:           s_setprio 3
; CHECK:           buffer_load_dwordx2
; CHECK-NEXT:      s_setprio 0
; CHECK-COUNT-4:   v_add_f32_e32
; CHECK:           s_cbranch_scc0 [[A:.*]]
; CHECK:       {{.*}}:  ; %b
; CHECK-NEXT:      buffer_load_dwordx2
; CHECK:           s_branch [[END:.*]]
; CHECK:       [[A]]:  ; %a
; CHECK:           s_branch [[END]]
; CHECK:       [[END]]:
define amdgpu_ps <2 x float> @valu_insts_threshold(<4 x i32> inreg %p, i32 inreg %i) "amdgpu-wave-priority-threshold"="4" {
  %v = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 0, i32 0)
  %add = fadd <2 x float> %v, %v
  %add2 = fadd <2 x float> %add, %add

  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %a, label %b

a:
  ret <2 x float> %add2

b:
  %v2 = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 1, i32 0, i32 0)
  %sub = fsub <2 x float> %add2, %v2
  ret <2 x float> %sub
}

; CHECK-LABEL: valu_insts_threshold2:
; CHECK-NOT: s_setprio
; CHECK: ; -- End function
define amdgpu_ps <2 x float> @valu_insts_threshold2(<4 x i32> inreg %p, i32 inreg %i) "amdgpu-wave-priority-threshold"="5" {
  %v = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 0, i32 0, i32 0)
  %add = fadd <2 x float> %v, %v
  %add2 = fadd <2 x float> %add, %add

  %cond = icmp eq i32 %i, 0
  br i1 %cond, label %a, label %b

a:
  ret <2 x float> %add2

b:
  %v2 = call <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32> %p, i32 0, i32 1, i32 0, i32 0)
  %sub = fsub <2 x float> %add2, %v2
  ret <2 x float> %sub
}

declare <2 x float> @llvm.amdgcn.struct.buffer.load.v2f32(<4 x i32>, i32, i32, i32, i32) nounwind
