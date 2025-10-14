; RUN: llc -mcpu=gfx942 < %s | FileCheck %s
; CHECK: v_add_f32_e32
; CHECK-NEXT: ;;#ASMSTART
; CHECK-NEXT: v_mfma_f64
; CHECK-NEXT: ;;#ASMEND
; CHECK: v_add_f32_e32
; ModuleID = '<stdin>'
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: convergent mustprogress norecurse nounwind
define protected amdgpu_kernel void @_Z17group4_sum_floaatPfPKfi(ptr addrspace(1) noalias noundef writeonly captures(none) %to.coerce, ptr addrspace(1) noalias noundef readonly captures(none) %from.coerce, i32 noundef %length) local_unnamed_addr #0 {
entry:
  %0 = tail call i32 @llvm.amdgcn.workgroup.id.x()
  %mul = shl i32 %0, 6
  %1 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %add = add i32 %mul, %1
  %cmp = icmp slt i32 %add, %length
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %idx.ext = sext i32 %add to i64
  %add.ptr = getelementptr inbounds float, ptr addrspace(1) %to.coerce, i64 %idx.ext
  %mul3 = shl nsw i32 %add, 2
  %idx.ext4 = sext i32 %mul3 to i64
  %add.ptr5 = getelementptr inbounds float, ptr addrspace(1) %from.coerce, i64 %idx.ext4
  %2 = load <2 x float>, ptr addrspace(1) %add.ptr5, align 16
  %a20 = add i64 %idx.ext4, 2
  %a21 = getelementptr inbounds float, ptr addrspace(1) %from.coerce, i64 %a20
  %a22 = load <2 x float>, ptr addrspace(1) %a21, align 16
  %3 = extractelement <2 x float> %a22, i64 1
  %4 = extractelement <2 x float> %2, i64 0
  %5 = tail call contract noundef float asm "v_mfma_f64_4x4x4f64 $0, $1, $2, 0", "=a,v,v"(<2 x float> %2, <2 x float> %a22) #3
  %6 = extractelement <2 x float> %2, i64 1
  %7 = extractelement <2 x float> %a22, i64 0
  %add6 = fadd contract float %6, %7
  %add7 = fadd contract float %5, %add6
  store float %add7, ptr addrspace(1) %add.ptr, align 4
  tail call void @llvm.amdgcn.sched.group.barrier(i32 16, i32 1, i32 0)
  tail call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 5, i32 0)
  tail call void @llvm.amdgcn.sched.group.barrier(i32 16, i32 1, i32 0)
  tail call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 1, i32 0)
  tail call void @llvm.amdgcn.sched.group.barrier(i32 8, i32 1, i32 0)
  tail call void @llvm.amdgcn.sched.group.barrier(i32 2, i32 1, i32 0)
  br label %if.end

if.end:                                           ; preds = %if.then, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #1

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.group.barrier(i32 immarg, i32 immarg, i32 immarg) #2

attributes #0 = { convergent mustprogress norecurse nounwind "amdgpu-agpr-alloc"="0" "amdgpu-flat-work-group-size"="1,1024" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-waves-per-eu"="4,8" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-fmin-fmax-global-f64,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,+xf32-insts" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { convergent nounwind memory(none) }
