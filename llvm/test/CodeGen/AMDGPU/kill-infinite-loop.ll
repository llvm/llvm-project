; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -enable-var-scope %s
; Although it's modeled without any control flow in order to get better code
; out of the structurizer, @llvm.amdgcn.kill actually ends the thread that calls
; it with "true". In case it's called in a provably infinite loop, we still
; need to successfully exit and export something, even if we can't know where
; to jump to in the LLVM IR. Therefore we insert a null export ourselves in
; this case right before the s_endpgm to avoid GPU hangs, which is what this
; tests.

; CHECK-LABEL: return_void
; Make sure that we remove the done bit from the original export
; CHECK: exp mrt0 v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}}, v{{[0-9]+}} vm
; CHECK: exp null off, off, off, off done vm
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @return_void(float %0) #0 {
main_body:
  %cmp = fcmp olt float %0, 1.000000e+01
  br i1 %cmp, label %end, label %loop

loop:
  call void @llvm.amdgcn.kill(i1 false) #3
  br label %loop

end:
  call void @llvm.amdgcn.exp.f32(i32 0, i32 15, float 0., float 0., float 0., float 1., i1 true, i1 true) #3
  ret void
}

; Check that we also remove the done bit from compressed exports correctly.
; CHECK-LABEL: return_void_compr
; CHECK: exp mrt0 v{{[0-9]+}}, off, v{{[0-9]+}}, off compr vm
; CHECK: exp null off, off, off, off done vm
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @return_void_compr(float %0) #0 {
main_body:
  %cmp = fcmp olt float %0, 1.000000e+01
  br i1 %cmp, label %end, label %loop

loop:
  call void @llvm.amdgcn.kill(i1 false) #3
  br label %loop

end:
  call void @llvm.amdgcn.exp.compr.v2i16(i32 0, i32 5, <2 x i16> < i16 0, i16 0 >, <2 x i16> < i16 0, i16 0 >, i1 true, i1 true) #3
  ret void
}

; test the case where there's only a kill in an infinite loop
; CHECK-LABEL: only_kill
; CHECK: exp null off, off, off, off done vm
; CHECK-NEXT: s_endpgm
; SIInsertSkips inserts an extra null export here, but it should be harmless.
; CHECK: exp null off, off, off, off done vm
; CHECK-NEXT: s_endpgm
define amdgpu_ps void @only_kill() #0 {
main_body:
  br label %loop

loop:
  call void @llvm.amdgcn.kill(i1 false) #3
  br label %loop
}

; Check that the epilog is the final block
; CHECK-LABEL: return_nonvoid
; CHECK: exp null off, off, off, off done vm
; CHECK-NEXT: s_endpgm
; CHECK-NEXT: BB{{[0-9]+}}_{{[0-9]+}}:
define amdgpu_ps float @return_nonvoid(float %0) #0 {
main_body:
  %cmp = fcmp olt float %0, 1.000000e+01
  br i1 %cmp, label %end, label %loop

loop:
  call void @llvm.amdgcn.kill(i1 false) #3
  br label %loop

end:
  ret float 0.
}

declare void @llvm.amdgcn.kill(i1) #0
declare void @llvm.amdgcn.exp.f32(i32 immarg, i32 immarg, float, float, float, float, i1 immarg, i1 immarg) #0
declare void @llvm.amdgcn.exp.compr.v2i16(i32 immarg, i32 immarg, <2 x i16>, <2 x i16>, i1 immarg, i1 immarg) #0

attributes #0 = { nounwind }
