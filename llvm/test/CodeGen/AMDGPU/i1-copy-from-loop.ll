; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn-- -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}i1_copy_from_loop:
;
; SI:   ; %Flow
; SI:  s_or_b64 [[EXIT_MASK:s\[[0-9]+:[0-9]+\]]]
; SI:  s_and_b64 [[ACCUM_MASK:s\[[0-9]+:[0-9]+\]]], [[CC_MASK:s\[[0-9]+:[0-9]+\]]], exec
; SI:  s_or_b64  [[I1_VALUE:s\[[0-9]+:[0-9]+\]]], {{s\[[0-9]+:[0-9]+\]}}, [[ACCUM_MASK]]
; SI:  s_cbranch_execz [[FOR_END_LABEL:BB0_[0-9]+]]

; SI: ; %for.body
; SI:      v_cmp_lt_u32_e64  [[CC_MASK]], s{{[0-9]+}}, 4

; SI: [[FOR_END_LABEL]]
; SI:      s_or_b64 exec, exec, [[EXIT_MASK]]
; SI:      s_and_saveexec_b64 {{s\[[0-9]+:[0-9]+\]}}, [[I1_VALUE]]
; SI:      s_cbranch_execz [[EXIT:BB0_[0-9]+]]
; SI: [[EXIT]]
; SI-NEXT: s_endpgm

define amdgpu_ps void @i1_copy_from_loop(<4 x i32> inreg %rsrc, i32 %tid) {
entry:
  br label %for.body

for.body:
  %i = phi i32 [0, %entry], [%i.inc, %end.loop]
  %cc = icmp ult i32 %i, 4
  br i1 %cc, label %mid.loop, label %for.end

mid.loop:
  %v = call float @llvm.amdgcn.buffer.load.f32(<4 x i32> %rsrc, i32 %tid, i32 %i, i1 false, i1 false)
  %cc2 = fcmp oge float %v, 0.0
  br i1 %cc2, label %end.loop, label %for.end

end.loop:
  %i.inc = add i32 %i, 1
  br label %for.body

for.end:
  br i1 %cc, label %if, label %end

if:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float undef, float undef, float undef, float undef, i1 true, i1 true)
  br label %end

end:
  ret void
}

declare float @llvm.amdgcn.buffer.load.f32(<4 x i32>, i32, i32, i1, i1) #0
declare void @llvm.amdgcn.exp.f32(i32, i32, float, float, float, float, i1, i1) #1

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind inaccessiblememonly }
