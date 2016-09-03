;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=SI
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s --check-prefix=CHECK --check-prefix=VI

; Check that WQM isn't triggered by image load/store intrinsics.
;
;CHECK-LABEL: {{^}}test1:
;CHECK-NOT: s_wqm
define amdgpu_ps <4 x float> @test1(<8 x i32> inreg %rsrc, <4 x i32> %c) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.v4i32(<4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  call void @llvm.amdgcn.image.store.v4i32(<4 x float> %tex, <4 x i32> %c, <8 x i32> %rsrc, i32 15, i1 0, i1 0, i1 0, i1 0)
  ret <4 x float> %tex
}

; Check that WQM is triggered by image samples and left untouched for loads...
;
;CHECK-LABEL: {{^}}test2:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK-NOT: exec
define amdgpu_ps void @test2(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float addrspace(1)* inreg %ptr, <4 x i32> %c) {
main_body:
  %c.1 = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %c, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %c.2 = bitcast <4 x float> %c.1 to <4 x i32>
  %c.3 = extractelement <4 x i32> %c.2, i32 0
  %gep = getelementptr float, float addrspace(1)* %ptr, i32 %c.3
  %data = load float, float addrspace(1)* %gep

  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float %data, float undef, float undef, float undef)

  ret void
}

; ... but disabled for stores (and, in this simple case, not re-enabled).
;
;CHECK-LABEL: {{^}}test3:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: image_sample
;CHECK: store
;CHECK-NOT: exec
;CHECK: .size test3
define amdgpu_ps <4 x float> @test3(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <4 x i32> %c) {
main_body:
  %tex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %c, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %tex.2 = extractelement <4 x i32> %tex.1, i32 0

  call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> %tex, <4 x i32> undef, i32 %tex.2, i32 0, i1 0, i1 0)

  ret <4 x float> %tex
}

; Check that WQM is re-enabled when required.
;
;CHECK-LABEL: {{^}}test4:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: v_mul_lo_i32 [[MUL:v[0-9]+]], v0, v1
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: store
;CHECK: s_wqm_b64 exec, exec
;CHECK: image_sample
;CHECK: image_sample
define amdgpu_ps <4 x float> @test4(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float addrspace(1)* inreg %ptr, i32 %c, i32 %d, float %data) {
main_body:
  %c.1 = mul i32 %c, %d

  call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> undef, <4 x i32> undef, i32 %c.1, i32 0, i1 0, i1 0)

  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %c.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret <4 x float> %dtex
}

; Check a case of one branch of an if-else requiring WQM, the other requiring
; exact.
;
; Note: In this particular case, the save-and-restore could be avoided if the
; analysis understood that the two branches of the if-else are mutually
; exclusive.
;
;CHECK-LABEL: {{^}}test_control_flow_0:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: %ELSE
;CHECK: s_and_saveexec_b64 [[SAVED:s\[[0-9]+:[0-9]+\]]], [[ORIG]]
;CHECK: store
;CHECK: s_mov_b64 exec, [[SAVED]]
;CHECK: %IF
;CHECK: image_sample
;CHECK: image_sample
define amdgpu_ps float @test_control_flow_0(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ELSE

IF:
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %c, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %data.if = extractelement <4 x float> %dtex, i32 0
  br label %END

ELSE:
  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 %c, i32 0, i1 0, i1 0)
  br label %END

END:
  %r = phi float [ %data.if, %IF ], [ %data, %ELSE ]
  ret float %r
}

; Reverse branch order compared to the previous test.
;
;CHECK-LABEL: {{^}}test_control_flow_1:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: %IF
;CHECK: image_sample
;CHECK: image_sample
;CHECK: %Flow
;CHECK-NEXT: s_or_saveexec_b64 [[SAVED:s\[[0-9]+:[0-9]+\]]],
;CHECK-NEXT: s_and_b64 exec, exec, [[ORIG]]
;CHECK-NEXT: s_and_b64 [[SAVED]], exec, [[SAVED]]
;CHECK-NEXT: s_xor_b64 exec, exec, [[SAVED]]
;CHECK-NEXT: mask branch [[END_BB:BB[0-9]+_[0-9]+]]
;CHECK-NEXT: BB{{[0-9]+_[0-9]+}}: ; %ELSE
;CHECK: store_dword
;CHECK: [[END_BB]]: ; %END
;CHECK: s_or_b64 exec, exec,
;CHECK: v_mov_b32_e32 v0
;CHECK: ; return
define amdgpu_ps float @test_control_flow_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %ELSE, label %IF

IF:
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %c, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %data.if = extractelement <4 x float> %dtex, i32 0
  br label %END

ELSE:
  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 %c, i32 0, i1 0, i1 0)
  br label %END

END:
  %r = phi float [ %data.if, %IF ], [ %data, %ELSE ]
  ret float %r
}

; Check that branch conditions are properly marked as needing WQM...
;
;CHECK-LABEL: {{^}}test_control_flow_2:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: store
;CHECK: s_wqm_b64 exec, exec
;CHECK: load
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: store
;CHECK: s_wqm_b64 exec, exec
;CHECK: v_cmp
define amdgpu_ps <4 x float> @test_control_flow_2(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <3 x i32> %idx, <2 x float> %data, i32 %coord) {
main_body:
  %idx.1 = extractelement <3 x i32> %idx, i32 0
  %data.1 = extractelement <2 x float> %data, i32 0
  call void @llvm.amdgcn.buffer.store.f32(float %data.1, <4 x i32> undef, i32 %idx.1, i32 0, i1 0, i1 0)

  ; The load that determines the branch (and should therefore be WQM) is
  ; surrounded by stores that require disabled WQM.
  %idx.2 = extractelement <3 x i32> %idx, i32 1
  %z = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx.2, i32 0, i1 0, i1 0)

  %idx.3 = extractelement <3 x i32> %idx, i32 2
  %data.3 = extractelement <2 x float> %data, i32 1
  call void @llvm.amdgcn.buffer.store.f32(float %data.3, <4 x i32> undef, i32 %idx.3, i32 0, i1 0, i1 0)

  %cc = fcmp ogt float %z, 0.0
  br i1 %cc, label %IF, label %ELSE

IF:
  %coord.IF = mul i32 %coord, 3
  br label %END

ELSE:
  %coord.ELSE = mul i32 %coord, 4
  br label %END

END:
  %coord.END = phi i32 [ %coord.IF, %IF ], [ %coord.ELSE, %ELSE ]
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %coord.END, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret <4 x float> %tex
}

; ... but only if they really do need it.
;
;CHECK-LABEL: {{^}}test_control_flow_3:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: image_sample
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: image_sample
;CHECK: store
;CHECK: v_cmp
define amdgpu_ps float @test_control_flow_3(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, i32 %coord) {
main_body:
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %dtex.1 = extractelement <4 x float> %dtex, i32 0

  call void @llvm.amdgcn.buffer.store.f32(float %dtex.1, <4 x i32> undef, i32 %idx, i32 0, i1 0, i1 0)

  %cc = fcmp ogt float %dtex.1, 0.0
  br i1 %cc, label %IF, label %ELSE

IF:
  %tex.IF = fmul float %dtex.1, 3.0
  br label %END

ELSE:
  %tex.ELSE = fmul float %dtex.1, 4.0
  br label %END

END:
  %tex.END = phi float [ %tex.IF, %IF ], [ %tex.ELSE, %ELSE ]
  ret float %tex.END
}

; Another test that failed at some point because of terminator handling.
;
;CHECK-LABEL: {{^}}test_control_flow_4:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: %IF
;CHECK: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]],  [[ORIG]]
;CHECK: load
;CHECK: store
;CHECK: s_mov_b64 exec, [[SAVE]]
;CHECK: %END
;CHECK: image_sample
;CHECK: image_sample
define amdgpu_ps <4 x float> @test_control_flow_4(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %coord, i32 %y, float %z) {
main_body:
  %cond = icmp eq i32 %y, 0
  br i1 %cond, label %IF, label %END

IF:
  %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 0, i32 0, i1 0, i1 0)
  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 1, i32 0, i1 0, i1 0)
  br label %END

END:
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret <4 x float> %dtex
}

; Kill is performed in WQM mode so that uniform kill behaves correctly ...
;
;CHECK-LABEL: {{^}}test_kill_0:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: image_sample
;CHECK: buffer_store_dword
;CHECK: s_wqm_b64 exec, exec
;CHECK: v_cmpx_
;CHECK: s_and_saveexec_b64 [[SAVE:s\[[0-9]+:[0-9]+\]]], [[ORIG]]
;CHECK: buffer_store_dword
;CHECK: s_mov_b64 exec, [[SAVE]]
;CHECK: image_sample
define amdgpu_ps <4 x float> @test_kill_0(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float addrspace(1)* inreg %ptr, <2 x i32> %idx, <2 x float> %data, i32 %coord, i32 %coord2, float %z) {
main_body:
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)

  %idx.0 = extractelement <2 x i32> %idx, i32 0
  %data.0 = extractelement <2 x float> %data, i32 0
  call void @llvm.amdgcn.buffer.store.f32(float %data.0, <4 x i32> undef, i32 %idx.0, i32 0, i1 0, i1 0)

  call void @llvm.AMDGPU.kill(float %z)

  %idx.1 = extractelement <2 x i32> %idx, i32 1
  %data.1 = extractelement <2 x float> %data, i32 1
  call void @llvm.amdgcn.buffer.store.f32(float %data.1, <4 x i32> undef, i32 %idx.1, i32 0, i1 0, i1 0)

  %tex2 = call <4 x float> @llvm.SI.image.sample.i32(i32 %coord2, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex2.1 = bitcast <4 x float> %tex2 to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex2.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %out = fadd <4 x float> %tex, %dtex

  ret <4 x float> %out
}

; ... but only if WQM is necessary.
;
; CHECK-LABEL: {{^}}test_kill_1:
; CHECK-NEXT: ; %main_body
; CHECK: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec
; CHECK: image_sample
; CHECK: s_and_b64 exec, exec, [[ORIG]]
; CHECK: image_sample
; CHECK: buffer_store_dword
; CHECK-NOT: wqm
; CHECK: v_cmpx_
define amdgpu_ps <4 x float> @test_kill_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, i32 %coord, i32 %coord2, float %z) {
main_body:
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.1, <8 x i32> %rsrc, <4 x i32> %sampler, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)

  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 0, i32 0, i1 0, i1 0)

  call void @llvm.AMDGPU.kill(float %z)

  ret <4 x float> %dtex
}

; Check prolog shaders.
;
; CHECK-LABEL: {{^}}test_prolog_1:
; CHECK: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec
; CHECK: v_add_f32_e32 v0,
; CHECK: s_and_b64 exec, exec, [[ORIG]]
define amdgpu_ps float @test_prolog_1(float %a, float %b) #4 {
main_body:
  %s = fadd float %a, %b
  ret float %s
}

; CHECK-LABEL: {{^}}test_loop_vcc:
; CHECK-NEXT: ; %entry
; CHECK-NEXT: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec
; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK: image_store
; CHECK: s_wqm_b64 exec, exec
; CHECK-DAG: v_mov_b32_e32 [[CTR:v[0-9]+]], 0
; CHECK-DAG: v_mov_b32_e32 [[SEVEN:v[0-9]+]], 0x40e00000
; CHECK: s_branch [[LOOPHDR:BB[0-9]+_[0-9]+]]

; CHECK: v_add_f32_e32 [[CTR]], 2.0, [[CTR]]
; CHECK: [[LOOPHDR]]: ; %loop
; CHECK: v_cmp_lt_f32_e32 vcc, [[SEVEN]], [[CTR]]
; CHECK: s_cbranch_vccz
; CHECK: ; %break

; CHECK: ; return
define amdgpu_ps <4 x float> @test_loop_vcc(<4 x float> %in) nounwind {
entry:
  call void @llvm.amdgcn.image.store.v4i32(<4 x float> %in, <4 x i32> undef, <8 x i32> undef, i32 15, i1 0, i1 0, i1 0, i1 0)
  br label %loop

loop:
  %ctr.iv = phi float [ 0.0, %entry ], [ %ctr.next, %body ]
  %c.iv = phi <4 x float> [ %in, %entry ], [ %c.next, %body ]
  %cc = fcmp ogt float %ctr.iv, 7.0
  br i1 %cc, label %break, label %body

body:
  %c.i = bitcast <4 x float> %c.iv to <4 x i32>
  %c.next = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %c.i, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %ctr.next = fadd float %ctr.iv, 2.0
  br label %loop

break:
  ret <4 x float> %c.iv
}

; Only intrinsic stores need exact execution -- other stores do not have
; externally visible effects and may require WQM for correctness.
;
; CHECK-LABEL: {{^}}test_alloca:
; CHECK: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec

; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, 0
; CHECK: s_wqm_b64 exec, exec
; CHECK: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen
; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 idxen
; CHECK: s_wqm_b64 exec, exec
; CHECK: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen

; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK: image_sample
; CHECK: buffer_store_dwordx4
define amdgpu_ps void @test_alloca(float %data, i32 %a, i32 %idx) nounwind {
entry:
  %array = alloca [32 x i32], align 4

  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 0, i32 0, i1 0, i1 0)

  %s.gep = getelementptr [32 x i32], [32 x i32]* %array, i32 0, i32 0
  store volatile i32 %a, i32* %s.gep, align 4

  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 1, i32 0, i1 0, i1 0)

  %c.gep = getelementptr [32 x i32], [32 x i32]* %array, i32 0, i32 %idx
  %c = load i32, i32* %c.gep, align 4

  %t = call <4 x float> @llvm.SI.image.sample.i32(i32 %c, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)

  call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> %t, <4 x i32> undef, i32 0, i32 0, i1 0, i1 0)

  ret void
}

; Must return to exact at the end of a non-void returning shader,
; otherwise the EXEC mask exported by the epilog will be wrong. This is true
; even if the shader has no kills, because a kill could have happened in a
; previous shader fragment.
;
; CHECK-LABEL: {{^}}test_nonvoid_return:
; CHECK: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec
;
; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK-NOT: exec
define amdgpu_ps <4 x float> @test_nonvoid_return() nounwind {
  %tex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.i = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.i, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  ret <4 x float> %dtex
}

; CHECK-LABEL: {{^}}test_nonvoid_return_unreachable:
; CHECK: s_mov_b64 [[LIVE:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec
;
; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK-NOT: exec
define amdgpu_ps <4 x float> @test_nonvoid_return_unreachable(i32 inreg %c) nounwind {
entry:
  %tex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> undef, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tex.i = bitcast <4 x float> %tex to <4 x i32>
  %dtex = call <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32> %tex.i, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)

  %cc = icmp sgt i32 %c, 0
  br i1 %cc, label %if, label %else

if:
  store volatile <4 x float> %dtex, <4 x float>* undef
  unreachable

else:
  ret <4 x float> %dtex
}

declare void @llvm.amdgcn.image.store.v4i32(<4 x float>, <4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #1
declare void @llvm.amdgcn.buffer.store.f32(float, <4 x i32>, i32, i32, i1, i1) #1
declare void @llvm.amdgcn.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i1, i1) #1

declare <4 x float> @llvm.amdgcn.image.load.v4i32(<4 x i32>, <8 x i32>, i32, i1, i1, i1, i1) #2
declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #2

declare <4 x float> @llvm.SI.image.sample.i32(i32, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #3
declare <4 x float> @llvm.SI.image.sample.v4i32(<4 x i32>, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #3

declare void @llvm.AMDGPU.kill(float)
declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }
attributes #3 = { nounwind readnone }
attributes #4 = { "amdgpu-ps-wqm-outputs" }
