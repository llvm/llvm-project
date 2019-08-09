; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=CHECK -check-prefix=VI %s

; Check that WQM isn't triggered by image load/store intrinsics.
;
;CHECK-LABEL: {{^}}test1:
;CHECK-NOT: s_wqm
define amdgpu_ps <4 x float> @test1(<8 x i32> inreg %rsrc, i32 %c) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32 15, i32 %c, <8 x i32> %rsrc, i32 0, i32 0)
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %tex, i32 15, i32 %c, <8 x i32> %rsrc, i32 0, i32 0)
  ret <4 x float> %tex
}

; Check that WQM is triggered by code calculating inputs to image samples and is disabled as soon as possible
;
;CHECK-LABEL: {{^}}test2:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: interp
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK-NOT: interp
;CHECK: image_sample
;CHECK-NOT: exec
;CHECK: .size test2
define amdgpu_ps <4 x float> @test2(i32 inreg, i32 inreg, i32 inreg, i32 inreg %m0, <8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <2 x float> %pos) #6 {
main_body:
  %inst23 = extractelement <2 x float> %pos, i32 0
  %inst24 = extractelement <2 x float> %pos, i32 1
  %inst25 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 0, i32 0, i32 %m0)
  %inst26 = tail call float @llvm.amdgcn.interp.p2(float %inst25, float %inst24, i32 0, i32 0, i32 %m0)
  %inst28 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 1, i32 0, i32 %m0)
  %inst29 = tail call float @llvm.amdgcn.interp.p2(float %inst28, float %inst24, i32 1, i32 0, i32 %m0)
  %tex = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %inst26, float %inst29, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  ret <4 x float> %tex
}

; ... but disabled for stores (and, in this simple case, not re-enabled) ...
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
define amdgpu_ps <4 x float> @test3(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float %c) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex.1 = bitcast <4 x float> %tex to <4 x i32>
  %tex.2 = extractelement <4 x i32> %tex.1, i32 0

  call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> %tex, <4 x i32> undef, i32 %tex.2, i32 0, i1 0, i1 0)

  ret <4 x float> %tex
}

; ... and disabled for export.
;
;CHECK-LABEL: {{^}}test3x:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: image_sample
;CHECK: exp
;CHECK-NOT: exec
;CHECK: .size test3x
define amdgpu_ps void @test3x(i32 inreg, i32 inreg, i32 inreg, i32 inreg %m0, <8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, <2 x float> %pos) #6 {
main_body:
  %inst23 = extractelement <2 x float> %pos, i32 0
  %inst24 = extractelement <2 x float> %pos, i32 1
  %inst25 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 0, i32 0, i32 %m0)
  %inst26 = tail call float @llvm.amdgcn.interp.p2(float %inst25, float %inst24, i32 0, i32 0, i32 %m0)
  %inst28 = tail call float @llvm.amdgcn.interp.p1(float %inst23, i32 1, i32 0, i32 %m0)
  %inst29 = tail call float @llvm.amdgcn.interp.p2(float %inst28, float %inst24, i32 1, i32 0, i32 %m0)
  %tex = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float %inst26, float %inst29, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex.0 = extractelement <4 x float> %tex, i32 0
  %tex.1 = extractelement <4 x float> %tex, i32 1
  %tex.2 = extractelement <4 x float> %tex, i32 2
  %tex.3 = extractelement <4 x float> %tex, i32 3
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float %tex.0, float %tex.1, float %tex.2, float %tex.3, i1 true, i1 true)
  ret void
}

; Check that WQM is re-enabled when required.
;
;CHECK-LABEL: {{^}}test4:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: v_mul_lo_u32 [[MUL:v[0-9]+]], v0, v1
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: store
;CHECK: s_wqm_b64 exec, exec
;CHECK: image_sample
;CHECK: image_sample
define amdgpu_ps <4 x float> @test4(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float addrspace(1)* inreg %ptr, i32 %c, i32 %d, float %data) {
main_body:
  %c.1 = mul i32 %c, %d

  call void @llvm.amdgcn.buffer.store.v4f32(<4 x float> undef, <4 x i32> undef, i32 %c.1, i32 0, i1 0, i1 0)
  %c.1.bc = bitcast i32 %c.1 to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.1.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  ret <4 x float> %dtex
}

; Check that WQM is triggered by the wqm intrinsic.
;
;CHECK-LABEL: {{^}}test5:
;CHECK: s_wqm_b64 exec, exec
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
define amdgpu_ps float @test5(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  %src1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wqm.f32(float %out)
  ret float %out.0
}

; Check that the wqm intrinsic works correctly for integers.
;
;CHECK-LABEL: {{^}}test6:
;CHECK: s_wqm_b64 exec, exec
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
define amdgpu_ps float @test6(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  %src1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  %out = fadd float %src0, %src1
  %out.0 = bitcast float %out to i32
  %out.1 = call i32 @llvm.amdgcn.wqm.i32(i32 %out.0)
  %out.2 = bitcast i32 %out.1 to float
  ret float %out.2
}

; Check that WWM is triggered by the wwm intrinsic.
;
;CHECK-LABEL: {{^}}test_wwm1:
;CHECK: s_or_saveexec_b64 s{{\[[0-9]+:[0-9]+\]}}, -1
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
define amdgpu_ps float @test_wwm1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  %src1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  ret float %out.0
}

; Same as above, but with an integer type.
;
;CHECK-LABEL: {{^}}test_wwm2:
;CHECK: s_or_saveexec_b64 s{{\[[0-9]+:[0-9]+\]}}, -1
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: v_add_{{[iu]}}32_e32
define amdgpu_ps float @test_wwm2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  %src1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  %src0.0 = bitcast float %src0 to i32
  %src1.0 = bitcast float %src1 to i32
  %out = add i32 %src0.0, %src1.0
  %out.0 = call i32 @llvm.amdgcn.wwm.i32(i32 %out)
  %out.1 = bitcast i32 %out.0 to float
  ret float %out.1
}

; Check that we don't leave WWM on for computations that don't require WWM,
; since that will lead clobbering things that aren't supposed to be clobbered
; in cases like this.
;
;CHECK-LABEL: {{^}}test_wwm3:
;CHECK: s_or_saveexec_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], -1
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
;CHECK: s_mov_b64 exec, [[ORIG]]
;CHECK: v_add_f32_e32
define amdgpu_ps float @test_wwm3(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 32
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx, i32 0, i1 0, i1 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  %out.1 = fadd float %src, %out.0
  br label %endif

endif:
  %out.2 = phi float [ %out.1, %if ], [ 0.0, %main_body ]
  ret float %out.2
}

; Check that WWM writes aren't coalesced with non-WWM writes, since the WWM
; write could clobber disabled channels in the non-WWM one.
;
;CHECK-LABEL: {{^}}test_wwm4:
;CHECK: s_or_saveexec_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], -1
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
;CHECK: s_mov_b64 exec, [[ORIG]]
;CHECK-NEXT: v_mov_b32_e32
define amdgpu_ps float @test_wwm4(i32 inreg %idx) {
main_body:
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 32
  br i1 %cc, label %endif, label %if

if:
  %src = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx, i32 0, i1 0, i1 0)
  %out = fadd float %src, %src
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

; Make sure the transition from Exact to WWM then WQM works properly.
;
;CHECK-LABEL: {{^}}test_wwm5:
;CHECK: buffer_load_dword
;CHECK: buffer_store_dword
;CHECK: s_or_saveexec_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], -1
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
;CHECK: s_mov_b64 exec, [[ORIG]]
;CHECK: s_wqm_b64 exec, exec
define amdgpu_ps float @test_wwm5(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  call void @llvm.amdgcn.buffer.store.f32(float %src0, <4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  %src1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  %temp = fadd float %src1, %src1
  %temp.0 = call float @llvm.amdgcn.wwm.f32(float %temp)
  %out = fadd float %temp.0, %temp.0
  %out.0 = call float @llvm.amdgcn.wqm.f32(float %out)
  ret float %out.0
}

; Check that WWM is turned on correctly across basic block boundaries.
; if..then..endif version
;
;CHECK-LABEL: {{^}}test_wwm6_then:
;CHECK: s_or_saveexec_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], -1
;SI-CHECK: buffer_load_dword
;VI-CHECK: flat_load_dword
;CHECK: s_mov_b64 exec, [[ORIG]]
;CHECK: %if
;CHECK: s_or_saveexec_b64 [[ORIG2:s\[[0-9]+:[0-9]+\]]], -1
;SI-CHECK: buffer_load_dword
;VI-CHECK: flat_load_dword
;CHECK: v_add_f32_e32
;CHECK: s_mov_b64 exec, [[ORIG2]]
define amdgpu_ps float @test_wwm6_then() {
main_body:
  %src0 = load volatile float, float addrspace(1)* undef
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  %cc = icmp uge i32 %hi, 32
  br i1 %cc, label %endif, label %if

if:
  %src1 = load volatile float, float addrspace(1)* undef
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  br label %endif

endif:
  %out.1 = phi float [ %out.0, %if ], [ 0.0, %main_body ]
  ret float %out.1
}

; Check that WWM is turned on correctly across basic block boundaries.
; loop version
;
;CHECK-LABEL: {{^}}test_wwm6_loop:
;CHECK: s_or_saveexec_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], -1
;SI-CHECK: buffer_load_dword
;VI-CHECK: flat_load_dword
;CHECK: s_mov_b64 exec, [[ORIG]]
;CHECK: %loop
;CHECK: s_or_saveexec_b64 [[ORIG2:s\[[0-9]+:[0-9]+\]]], -1
;SI-CHECK: buffer_load_dword
;VI-CHECK: flat_load_dword
;CHECK: s_mov_b64 exec, [[ORIG2]]
define amdgpu_ps float @test_wwm6_loop() {
main_body:
  %src0 = load volatile float, float addrspace(1)* undef
  ; use mbcnt to make sure the branch is divergent
  %lo = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %hi = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %lo)
  br label %loop

loop:
  %counter = phi i32 [ %lo, %main_body ], [ %counter.1, %loop ]
  %src1 = load volatile float, float addrspace(1)* undef
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.wwm.f32(float %out)
  %counter.1 = sub i32 %counter, 1
  %cc = icmp ne i32 %counter.1, 0
  br i1 %cc, label %loop, label %endloop

endloop:
  ret float %out.0
}

; Check that @llvm.amdgcn.set.inactive disables WWM.
;
;CHECK-LABEL: {{^}}test_set_inactive1:
;CHECK: buffer_load_dword
;CHECK: s_not_b64 exec, exec
;CHECK: v_mov_b32_e32
;CHECK: s_not_b64 exec, exec
;CHECK: s_or_saveexec_b64 s{{\[[0-9]+:[0-9]+\]}}, -1
;CHECK: v_add_{{[iu]}}32_e32
define amdgpu_ps void @test_set_inactive1(i32 inreg %idx) {
main_body:
  %src = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx, i32 0, i1 0, i1 0)
  %src.0 = bitcast float %src to i32
  %src.1 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %src.0, i32 0)
  %out = add i32 %src.1, %src.1
  %out.0 = call i32 @llvm.amdgcn.wwm.i32(i32 %out)
  %out.1 = bitcast i32 %out.0 to float
  call void @llvm.amdgcn.buffer.store.f32(float %out.1, <4 x i32> undef, i32 %idx, i32 0, i1 0, i1 0)
  ret void
}

; Check that enabling WQM anywhere enables WQM for the set.inactive source.
;
;CHECK-LABEL: {{^}}test_set_inactive2:
;CHECK: s_wqm_b64 exec, exec
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
define amdgpu_ps void @test_set_inactive2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src1 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  %src1.0 = bitcast float %src1 to i32
  %src1.1 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %src1.0, i32 undef)
  %src0 = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i1 0, i1 0)
  %src0.0 = bitcast float %src0 to i32
  %src0.1 = call i32 @llvm.amdgcn.wqm.i32(i32 %src0.0)
  %out = add i32 %src0.1, %src1.1
  %out.0 = bitcast i32 %out to float
  call void @llvm.amdgcn.buffer.store.f32(float %out.0, <4 x i32> undef, i32 %idx1, i32 0, i1 0, i1 0)
  ret void
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
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
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
;CHECK-NEXT: s_cbranch_execz [[END_BB]]
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
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
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
  %coord.END.bc = bitcast i32 %coord.END to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord.END.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
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
;CHECK-DAG: v_cmp
;CHECK-DAG: store
define amdgpu_ps float @test_control_flow_3(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %coord) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
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
define amdgpu_ps <4 x float> @test_control_flow_4(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float %coord, i32 %y, float %z) {
main_body:
  %cond = icmp eq i32 %y, 0
  br i1 %cond, label %IF, label %END

IF:
  %data = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> undef, i32 0, i32 0, i1 0, i1 0)
  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 1, i32 0, i1 0, i1 0)
  br label %END

END:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
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
define amdgpu_ps <4 x float> @test_kill_0(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, float addrspace(1)* inreg %ptr, <2 x i32> %idx, <2 x float> %data, float %coord, float %coord2, float %z) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %idx.0 = extractelement <2 x i32> %idx, i32 0
  %data.0 = extractelement <2 x float> %data, i32 0
  call void @llvm.amdgcn.buffer.store.f32(float %data.0, <4 x i32> undef, i32 %idx.0, i32 0, i1 0, i1 0)

  %z.cmp = fcmp olt float %z, 0.0
  call void @llvm.amdgcn.kill(i1 %z.cmp)

  %idx.1 = extractelement <2 x i32> %idx, i32 1
  %data.1 = extractelement <2 x float> %data, i32 1
  call void @llvm.amdgcn.buffer.store.f32(float %data.1, <4 x i32> undef, i32 %idx.1, i32 0, i1 0, i1 0)
  %tex2 = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord2, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex2.0 = extractelement <4 x float> %tex2, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex2.0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
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
define amdgpu_ps <4 x float> @test_kill_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %idx, float %data, float %coord, float %coord2, float %z) {
main_body:
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %coord, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0

  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 0, i32 0, i1 0, i1 0)

  %z.cmp = fcmp olt float %z, 0.0
  call void @llvm.amdgcn.kill(i1 %z.cmp)

  ret <4 x float> %dtex
}

; Check prolog shaders.
;
; CHECK-LABEL: {{^}}test_prolog_1:
; CHECK: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec
; CHECK: v_add_f32_e32 v0,
; CHECK: s_and_b64 exec, exec, [[ORIG]]
define amdgpu_ps float @test_prolog_1(float %a, float %b) #5 {
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
; CHECK-DAG: s_mov_b32 [[SEVEN:s[0-9]+]], 0x40e00000

; CHECK: [[LOOPHDR:BB[0-9]+_[0-9]+]]: ; %body
; CHECK: v_add_f32_e32 [[CTR]], 2.0, [[CTR]]
; CHECK: v_cmp_lt_f32_e32 vcc, [[SEVEN]], [[CTR]]
; CHECK: s_cbranch_vccz [[LOOPHDR]]
; CHECK: ; %break

; CHECK: ; return
define amdgpu_ps <4 x float> @test_loop_vcc(<4 x float> %in) nounwind {
entry:
  call void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float> %in, i32 15, i32 undef, <8 x i32> undef, i32 0, i32 0)
  br label %loop

loop:
  %ctr.iv = phi float [ 0.0, %entry ], [ %ctr.next, %body ]
  %c.iv = phi <4 x float> [ %in, %entry ], [ %c.next, %body ]
  %cc = fcmp ogt float %ctr.iv, 7.0
  br i1 %cc, label %break, label %body

body:
  %c.iv0 = extractelement <4 x float> %c.iv, i32 0
  %c.next = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.iv0, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
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
; CHECK: buffer_store_dword {{v[0-9]+}}, off, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offset:4{{$}}
; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK: buffer_store_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, 0 idxen
; CHECK: s_wqm_b64 exec, exec
; CHECK: buffer_load_dword {{v[0-9]+}}, {{v[0-9]+}}, {{s\[[0-9]+:[0-9]+\]}}, {{s[0-9]+}} offen

; CHECK: s_and_b64 exec, exec, [[LIVE]]
; CHECK: image_sample
; CHECK: buffer_store_dwordx4
define amdgpu_ps void @test_alloca(float %data, i32 %a, i32 %idx) nounwind {
entry:
  %array = alloca [32 x i32], align 4, addrspace(5)

  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 0, i32 0, i1 0, i1 0)

  %s.gep = getelementptr [32 x i32], [32 x i32] addrspace(5)* %array, i32 0, i32 0
  store volatile i32 %a, i32 addrspace(5)* %s.gep, align 4

  call void @llvm.amdgcn.buffer.store.f32(float %data, <4 x i32> undef, i32 1, i32 0, i1 0, i1 0)

  %c.gep = getelementptr [32 x i32], [32 x i32] addrspace(5)* %array, i32 0, i32 %idx
  %c = load i32, i32 addrspace(5)* %c.gep, align 4
  %c.bc = bitcast i32 %c to float
  %t = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
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
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float undef, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
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
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float undef, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
  %cc = icmp sgt i32 %c, 0
  br i1 %cc, label %if, label %else

if:
  store volatile <4 x float> %dtex, <4 x float> addrspace(1)* undef
  unreachable

else:
  ret <4 x float> %dtex
}

; Test awareness that s_wqm_b64 clobbers SCC.
;
; CHECK-LABEL: {{^}}test_scc:
; CHECK: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: s_wqm_b64 exec, exec
; CHECK: s_cmp_
; CHECK-NEXT: s_cbranch_scc
; CHECK: ; %if
; CHECK: s_and_b64 exec, exec, [[ORIG]]
; CHECK: image_sample
; CHECK: ; %else
; CHECK: s_and_b64 exec, exec, [[ORIG]]
; CHECK: image_sample
; CHECK: ; %end
define amdgpu_ps <4 x float> @test_scc(i32 inreg %sel, i32 %idx) #1 {
main_body:
  %cc = icmp sgt i32 %sel, 0
  br i1 %cc, label %if, label %else

if:
  %r.if = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float 0.0, <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
  br label %end

else:
  %r.else = call <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32 15, float 0.0, float bitcast (i32 1 to float), <8 x i32> undef, <4 x i32> undef, i1 0, i32 0, i32 0) #0
  br label %end

end:
  %r = phi <4 x float> [ %r.if, %if ], [ %r.else, %else ]
  call void @llvm.amdgcn.buffer.store.f32(float 1.0, <4 x i32> undef, i32 %idx, i32 0, i1 0, i1 0)
  ret <4 x float> %r
}

; Check a case of a block being entirely WQM except for a bit of WWM.
; There was a bug where it forgot to enter and leave WWM.
;
;CHECK-LABEL: {{^}}test_wwm_within_wqm:
;CHECK: %IF
;CHECK: s_or_saveexec_b64 {{.*}}, -1
;CHECK: ds_swizzle
;
define amdgpu_ps float @test_wwm_within_wqm(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 %c, i32 %z, float %data) {
main_body:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ENDIF

IF:
  %dataf = extractelement <4 x float> %dtex, i32 0
  %data1 = fptosi float %dataf to i32
  %data2 = call i32 @llvm.amdgcn.set.inactive.i32(i32 %data1, i32 0)
  %data3 = call i32 @llvm.amdgcn.ds.swizzle(i32 %data2, i32 2079)
  %data4 = call i32 @llvm.amdgcn.wwm.i32(i32 %data3)
  %data4f = sitofp i32 %data4 to float
  br label %ENDIF

ENDIF:
  %r = phi float [ %data4f, %IF ], [ 0.0, %main_body ]
  ret float %r
}

declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #1
declare void @llvm.amdgcn.image.store.1d.v4f32.i32(<4 x float>, i32, i32, <8 x i32>, i32, i32) #1
declare void @llvm.amdgcn.buffer.store.f32(float, <4 x i32>, i32, i32, i1, i1) #2
declare void @llvm.amdgcn.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i1, i1) #2
declare <4 x float> @llvm.amdgcn.image.load.1d.v4f32.i32(i32, i32, <8 x i32>, i32, i32) #3
declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #3
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3
declare void @llvm.amdgcn.kill(i1) #1
declare float @llvm.amdgcn.wqm.f32(float) #3
declare i32 @llvm.amdgcn.wqm.i32(i32) #3
declare float @llvm.amdgcn.wwm.f32(float) #3
declare i32 @llvm.amdgcn.wwm.i32(i32) #3
declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32) #4
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3
declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #3
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #1
declare float @llvm.amdgcn.interp.p1(float, i32, i32, i32) #2
declare float @llvm.amdgcn.interp.p2(float, float, i32, i32, i32) #2
declare i32 @llvm.amdgcn.ds.swizzle(i32, i32)

attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }
attributes #3 = { nounwind readnone }
attributes #4 = { nounwind readnone convergent }
attributes #5 = { "amdgpu-ps-wqm-outputs" }
attributes #6 = { nounwind "InitialPSInputAddr"="2" }
