; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -verify-machineinstrs -O0 < %s

; GCN-LABEL: {{^}}test_loop:
; GCN: [[LABEL:BB[0-9+]_[0-9]+]]: ; %for.body{{$}}
; GCN: ds_read_b32
; GCN: ds_write_b32
; GCN: s_branch [[LABEL]]
; GCN: s_endpgm
define void @test_loop(float addrspace(3)* %ptr, i32 %n) nounwind {
entry:
  %cmp = icmp eq i32 %n, -1
  br i1 %cmp, label %for.exit, label %for.body

for.exit:
  ret void

for.body:
  %indvar = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %tmp = add i32 %indvar, 32
  %arrayidx = getelementptr float, float addrspace(3)* %ptr, i32 %tmp
  %vecload = load float, float addrspace(3)* %arrayidx, align 4
  %add = fadd float %vecload, 1.0
  store float %add, float addrspace(3)* %arrayidx, align 8
  %inc = add i32 %indvar, 1
  br label %for.body
}

; GCN-LABEL: @loop_const_true
; GCN: [[LABEL:BB[0-9+]_[0-9]+]]:
; GCN: ds_read_b32
; GCN: ds_write_b32
; GCN: s_branch [[LABEL]]
define void @loop_const_true(float addrspace(3)* %ptr, i32 %n) nounwind {
entry:
  br label %for.body

for.exit:
  ret void

for.body:
  %indvar = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %tmp = add i32 %indvar, 32
  %arrayidx = getelementptr float, float addrspace(3)* %ptr, i32 %tmp
  %vecload = load float, float addrspace(3)* %arrayidx, align 4
  %add = fadd float %vecload, 1.0
  store float %add, float addrspace(3)* %arrayidx, align 8
  %inc = add i32 %indvar, 1
  br i1 true, label %for.body, label %for.exit
}

; GCN-LABEL: {{^}}loop_const_false:
; GCN-NOT: s_branch
; GCN: s_endpgm
define void @loop_const_false(float addrspace(3)* %ptr, i32 %n) nounwind {
entry:
  br label %for.body

for.exit:
  ret void

; XXX - Should there be an S_ENDPGM?
for.body:
  %indvar = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %tmp = add i32 %indvar, 32
  %arrayidx = getelementptr float, float addrspace(3)* %ptr, i32 %tmp
  %vecload = load float, float addrspace(3)* %arrayidx, align 4
  %add = fadd float %vecload, 1.0
  store float %add, float addrspace(3)* %arrayidx, align 8
  %inc = add i32 %indvar, 1
  br i1 false, label %for.body, label %for.exit
}

; GCN-LABEL: {{^}}loop_const_undef:
; GCN-NOT: s_branch
; GCN: s_endpgm
define void @loop_const_undef(float addrspace(3)* %ptr, i32 %n) nounwind {
entry:
  br label %for.body

for.exit:
  ret void

; XXX - Should there be an s_endpgm?
for.body:
  %indvar = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %tmp = add i32 %indvar, 32
  %arrayidx = getelementptr float, float addrspace(3)* %ptr, i32 %tmp
  %vecload = load float, float addrspace(3)* %arrayidx, align 4
  %add = fadd float %vecload, 1.0
  store float %add, float addrspace(3)* %arrayidx, align 8
  %inc = add i32 %indvar, 1
  br i1 undef, label %for.body, label %for.exit
}

; GCN-LABEL: {{^}}loop_arg_0:
; GCN: v_and_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; GCN: v_cmp_eq_u32_e32 vcc, 1,

; GCN: [[LOOPBB:BB[0-9]+_[0-9]+]]
; GCN: s_add_i32 s{{[0-9]+}}, s{{[0-9]+}}, 0x80
; GCN: s_add_i32 s{{[0-9]+}}, s{{[0-9]+}}, 4

; GCN: s_cbranch_vccnz [[LOOPBB]]
; GCN-NEXT: ; BB#2
; GCN-NEXT: s_endpgm
define void @loop_arg_0(float addrspace(3)* %ptr, i32 %n, i1 %cond) nounwind {
entry:
  br label %for.body

for.exit:
  ret void

for.body:
  %indvar = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %tmp = add i32 %indvar, 32
  %arrayidx = getelementptr float, float addrspace(3)* %ptr, i32 %tmp
  %vecload = load float, float addrspace(3)* %arrayidx, align 4
  %add = fadd float %vecload, 1.0
  store float %add, float addrspace(3)* %arrayidx, align 8
  %inc = add i32 %indvar, 1
  br i1 %cond, label %for.body, label %for.exit
}
