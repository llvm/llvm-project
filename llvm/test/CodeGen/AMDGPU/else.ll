; RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

; CHECK-LABEL: {{^}}else_no_execfix:
; CHECK: ; %Flow
; CHECK-NEXT: s_or_saveexec_b64 [[DST:s\[[0-9]+:[0-9]+\]]],
; CHECK-NEXT: s_xor_b64 exec, exec, [[DST]]
; CHECK-NEXT: ; mask branch
define amdgpu_ps float @else_no_execfix(i32 %z, float %v) {
main_body:
  %cc = icmp sgt i32 %z, 5
  br i1 %cc, label %if, label %else

if:
  %v.if = fmul float %v, 2.0
  br label %end

else:
  %v.else = fmul float %v, 3.0
  br label %end

end:
  %r = phi float [ %v.if, %if ], [ %v.else, %else ]
  ret float %r
}

; CHECK-LABEL: {{^}}else_execfix_leave_wqm:
; CHECK: ; BB#0:
; CHECK-NEXT: s_mov_b64 [[INIT_EXEC:s\[[0-9]+:[0-9]+\]]], exec
; CHECK: ; %Flow
; CHECK-NEXT: s_or_saveexec_b64 [[DST:s\[[0-9]+:[0-9]+\]]],
; CHECK-NEXT: s_and_b64 exec, exec, [[INIT_EXEC]]
; CHECK-NEXT: s_and_b64 [[AND_INIT:s\[[0-9]+:[0-9]+\]]], exec, [[DST]]
; CHECK-NEXT: s_xor_b64 exec, exec, [[AND_INIT]]
; CHECK-NEXT: ; mask branch
define amdgpu_ps void @else_execfix_leave_wqm(i32 %z, float %v) {
main_body:
  %cc = icmp sgt i32 %z, 5
  br i1 %cc, label %if, label %else

if:
  %v.if = fmul float %v, 2.0
  br label %end

else:
  %c = fmul float %v, 3.0
  %c.i = bitcast float %c to i32
  %tex = call <4 x float> @llvm.SI.image.sample.i32(i32 %c.i, <8 x i32> undef, <4 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %v.else = extractelement <4 x float> %tex, i32 0
  br label %end

end:
  %r = phi float [ %v.if, %if ], [ %v.else, %else ]
  call void @llvm.amdgcn.buffer.store.f32(float %r, <4 x i32> undef, i32 0, i32 0, i1 0, i1 0)
  ret void
}

declare void @llvm.amdgcn.buffer.store.f32(float, <4 x i32>, i32, i32, i1, i1) nounwind

declare <4 x float> @llvm.SI.image.sample.i32(i32, <8 x i32>, <4 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) nounwind readnone
