; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=CHECK %s

; Check that WQM is not triggered by the softwqm intrinsic alone.
;
;CHECK-LABEL: {{^}}test1:
;CHECK-NOT: s_wqm_b64 exec, exec
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
define amdgpu_ps float @test1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = call float @llvm.amdgcn.softwqm.f32(float %out)
  ret float %out.0
}

; Check that the softwqm intrinsic works correctly for integers.
;
;CHECK-LABEL: {{^}}test2:
;CHECK-NOT: s_wqm_b64 exec, exec
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
define amdgpu_ps float @test2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %out.0 = bitcast float %out to i32
  %out.1 = call i32 @llvm.amdgcn.softwqm.i32(i32 %out.0)
  %out.2 = bitcast i32 %out.1 to float
  ret float %out.2
}

; Make sure the transition from WQM to Exact to softwqm does not trigger WQM.
;
;CHECK-LABEL: {{^}}test_softwqm1:
;CHECK-NOT: s_wqm_b64 exec, exec
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: buffer_store_dword
;CHECK-NOT; s_wqm_b64 exec, exec
;CHECK: v_add_f32_e32
define amdgpu_ps float @test_softwqm1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %src0, %src1
  call void @llvm.amdgcn.struct.buffer.store.f32(float %temp, <4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %out = fadd float %temp, %temp
  %out.0 = call float @llvm.amdgcn.softwqm.f32(float %out)
  ret float %out.0
}

; Make sure the transition from WQM to Exact to softwqm does trigger WQM.
;
;CHECK-LABEL: {{^}}test_softwqm2:
;CHECK: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK: s_wqm_b64 exec, exec
;CHECK: buffer_load_dword
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
;CHECK: v_add_f32_e32
;CHECK: s_and_b64 exec, exec, [[ORIG]]
;CHECK: buffer_store_dword
define amdgpu_ps float @test_softwqm2(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %src0, %src1
  %temp.0 = call float @llvm.amdgcn.wqm.f32(float %temp)
  call void @llvm.amdgcn.struct.buffer.store.f32(float %temp.0, <4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %out = fadd float %temp, %temp
  %out.0 = call float @llvm.amdgcn.softwqm.f32(float %out)
  ret float %out.0
}

; NOTE: llvm.amdgcn.wwm is deprecated, use llvm.amdgcn.strict.wwm instead.
; Make sure the transition from Exact to STRICT_WWM then softwqm does not trigger WQM.
;
;CHECK-LABEL: {{^}}test_wwm1:
;CHECK: s_or_saveexec_b64 [[ORIG0:s\[[0-9]+:[0-9]+\]]], -1
;CHECK: buffer_load_dword
;CHECK: s_mov_b64 exec, [[ORIG0]]
;CHECK: buffer_store_dword
;CHECK: s_or_saveexec_b64 [[ORIG1:s\[[0-9]+:[0-9]+\]]], -1
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
;CHECK: s_mov_b64 exec, [[ORIG1]]
;CHECK-NOT: s_wqm_b64
define amdgpu_ps float @test_wwm1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.struct.buffer.store.f32(float %src0, <4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %src0, %src1
  %temp.0 = call float @llvm.amdgcn.wwm.f32(float %temp)
  %out = fadd float %temp.0, %temp.0
  %out.0 = call float @llvm.amdgcn.softwqm.f32(float %out)
  ret float %out.0
}

; Make sure the transition from Exact to STRICT_WWM then softwqm does not trigger WQM.
;
;CHECK-LABEL: {{^}}test_strict_wwm1:
;CHECK: s_or_saveexec_b64 [[ORIG0:s\[[0-9]+:[0-9]+\]]], -1
;CHECK: buffer_load_dword
;CHECK: s_mov_b64 exec, [[ORIG0]]
;CHECK: buffer_store_dword
;CHECK: s_or_saveexec_b64 [[ORIG1:s\[[0-9]+:[0-9]+\]]], -1
;CHECK: buffer_load_dword
;CHECK: v_add_f32_e32
;CHECK: s_mov_b64 exec, [[ORIG1]]
;CHECK-NOT: s_wqm_b64
define amdgpu_ps float @test_strict_wwm1(i32 inreg %idx0, i32 inreg %idx1) {
main_body:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.struct.buffer.store.f32(float %src0, <4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %temp = fadd float %src0, %src1
  %temp.0 = call float @llvm.amdgcn.strict.wwm.f32(float %temp)
  %out = fadd float %temp.0, %temp.0
  %out.0 = call float @llvm.amdgcn.softwqm.f32(float %out)
  ret float %out.0
}


; Check that softwqm on one case of branch does not trigger WQM for shader.
;
;CHECK-LABEL: {{^}}test_control_flow_0:
;CHECK-NEXT: ; %main_body
;CHECK-NOT: s_wqm_b64 exec, exec
;CHECK: %ELSE
;CHECK: store
;CHECK: %IF
;CHECK: buffer_load
;CHECK: buffer_load
define amdgpu_ps float @test_control_flow_0(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 inreg %idx0, i32 inreg %idx1, i32 %c, i32 %z, float %data) {
main_body:
  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ELSE

IF:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %data.if = call float @llvm.amdgcn.softwqm.f32(float %out)
  br label %END

ELSE:
  call void @llvm.amdgcn.struct.buffer.store.f32(float %data, <4 x i32> undef, i32 %c, i32 0, i32 0, i32 0)
  br label %END

END:
  %r = phi float [ %data.if, %IF ], [ %data, %ELSE ]
  ret float %r
}

; Check that softwqm on one case of branch is treated as WQM in WQM shader.
;
;CHECK-LABEL: {{^}}test_control_flow_1:
;CHECK-NEXT: ; %main_body
;CHECK-NEXT: s_mov_b64 [[ORIG:s\[[0-9]+:[0-9]+\]]], exec
;CHECK-NEXT: s_wqm_b64 exec, exec
;CHECK: %ELSE
;CHECK: s_and_saveexec_b64 [[SAVED:s\[[0-9]+:[0-9]+\]]], [[ORIG]]
;CHECK: store
;CHECK: s_mov_b64 exec, [[SAVED]]
;CHECK: %IF
;CHECK-NOT: s_and_saveexec_b64
;CHECK-NOT: s_and_b64 exec
;CHECK: buffer_load
;CHECK: buffer_load
define amdgpu_ps float @test_control_flow_1(<8 x i32> inreg %rsrc, <4 x i32> inreg %sampler, i32 inreg %idx0, i32 inreg %idx1, i32 %c, i32 %z, float %data) {
main_body:
  %c.bc = bitcast i32 %c to float
  %tex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %c.bc, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %tex0 = extractelement <4 x float> %tex, i32 0
  %dtex = call <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32 15, float %tex0, <8 x i32> %rsrc, <4 x i32> %sampler, i1 0, i32 0, i32 0) #0
  %data.sample = extractelement <4 x float> %dtex, i32 0

  %cmp = icmp eq i32 %z, 0
  br i1 %cmp, label %IF, label %ELSE

IF:
  %src0 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx0, i32 0, i32 0, i32 0)
  %src1 = call float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32> undef, i32 %idx1, i32 0, i32 0, i32 0)
  %out = fadd float %src0, %src1
  %data.if = call float @llvm.amdgcn.softwqm.f32(float %out)
  br label %END

ELSE:
  call void @llvm.amdgcn.struct.buffer.store.f32(float %data.sample, <4 x i32> undef, i32 %c, i32 0, i32 0, i32 0)
  br label %END

END:
  %r = phi float [ %data.if, %IF ], [ %data, %ELSE ]
  ret float %r
}

declare void @llvm.amdgcn.struct.buffer.store.f32(float, <4 x i32>, i32, i32, i32, i32 immarg) #2
declare void @llvm.amdgcn.struct.buffer.store.v4f32(<4 x float>, <4 x i32>, i32, i32, i32, i32 immarg) #2
declare float @llvm.amdgcn.struct.buffer.load.f32(<4 x i32>, i32, i32, i32, i32 immarg) #3
declare <4 x float> @llvm.amdgcn.image.sample.1d.v4f32.f32(i32, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3
declare <4 x float> @llvm.amdgcn.image.sample.2d.v4f32.f32(i32, float, float, <8 x i32>, <4 x i32>, i1, i32, i32) #3
declare void @llvm.amdgcn.kill(i1) #1
declare float @llvm.amdgcn.wqm.f32(float) #3
declare float @llvm.amdgcn.softwqm.f32(float) #3
declare i32 @llvm.amdgcn.softwqm.i32(i32) #3
declare float @llvm.amdgcn.strict.wwm.f32(float) #3
declare float @llvm.amdgcn.wwm.f32(float) #3

attributes #1 = { nounwind }
attributes #2 = { nounwind readonly }
attributes #3 = { nounwind readnone }
