; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s
; RUN: not llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -vgpr-regalloc=basic -filetype=null %s 2>&1 | FileCheck -check-prefix=ERR %s

; ERR: error: <unknown>:0:0: no registers from class available to allocate in function 'no_free_vgprs_at_agpr_to_agpr_copy'

define void @no_free_vgprs_at_agpr_to_agpr_copy(float %v0, float %v1) #0 {
  %asm = call { <32 x i32>, <16 x float> } asm sideeffect "; def $0 $1", "=${v[0:31]},=${a[0:15]}"()
  %vgpr0 = extractvalue { <32 x i32>, <16 x float> } %asm, 0
  %agpr0 = extractvalue { <32 x i32>, <16 x float> } %asm, 1
  %mfma = call <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float %v0, float %v1, <16 x float> %agpr0, i32 0, i32 0, i32 0)
  %agpr1 = call i32 asm sideeffect "; copy ", "={a1},a,~{a[0:15]}"(<16 x float> %agpr0)
  %agpr2 = call i32 asm sideeffect "; copy ", "={a2},a,{a[0:15]}"(i32 %agpr1, <16 x float> %mfma)
  call void asm sideeffect "; use $0 $1", "{a3},{v[0:31]}"(i32 %agpr2, <32 x i32> %vgpr0)
  ret void
}

declare <16 x float> @llvm.amdgcn.mfma.f32.16x16x1f32(float, float, <16 x float>, i32 immarg, i32 immarg, i32 immarg) #1
declare noundef i32 @llvm.amdgcn.workitem.id.x() #2

attributes #0 = { "amdgpu-agpr-alloc"="0" "amdgpu-waves-per-eu"="6,6" }
attributes #1 = { convergent nocallback nofree nosync nounwind willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
