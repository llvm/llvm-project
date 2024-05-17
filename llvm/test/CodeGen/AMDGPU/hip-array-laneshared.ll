; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -mattr=-promote-alloca -verify-machineinstrs -o - %s | FileCheck %s

target datalayout = "A5"

@exchange = external addrspace(10) global [1000 x float], align 4
@__hip_cuid_ = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__hip_cuid_ to ptr)], section "llvm.metadata"
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define amdgpu_kernel void @_Z3foov() #0 {
entry:
; CHECK: s_getreg_b32 s33, hwreg(HW_REG_WAVE_GROUP_INFO, 16, 3)
; CHECK: s_mul_i32 s33, s33, 4
; CHECK: s_set_gpr_idx_u32 idx0, s33
; CHECK: s_getreg_b32 s33, hwreg(HW_REG_WAVE_GROUP_INFO, 16, 3)
; CHECK: s_mul_i32 s33, s33, 64
; CHECK: s_add_co_i32 s33, s33, 0xfa0

  %array.ascast = alloca [10 x float], align 4, addrspace(5)
  %i.ascast = alloca i32, align 4, addrspace(5)
  %i3.ascast = alloca i32, align 4, addrspace(5)
  store i32 0, ptr addrspace(5) %i.ascast, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr addrspace(5) %i.ascast, align 4
  %cmp = icmp slt i32 %0, 1000
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr addrspace(5) %i.ascast, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1000 x float], ptr addrspace(10) @exchange, i64 0, i64 %idxprom
; CHECK: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
  %2 = load float, ptr addrspace(10) %arrayidx, align 4
  %3 = load i32, ptr addrspace(5) %i.ascast, align 4
  %rem = srem i32 %3, 10
  %idxprom1 = sext i32 %rem to i64
  %arrayidx2 = getelementptr inbounds [10 x float], ptr addrspace(5) %array.ascast, i64 0, i64 %idxprom1
; CHECK: 	scratch_store_b32 {{v[0-9]+}}, {{v[0-9]+}}, s33 scale_offset
  store float %2, ptr addrspace(5) %arrayidx2, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr addrspace(5) %i.ascast, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr addrspace(5) %i.ascast, align 4
  br label %for.cond, !llvm.loop !4

for.end:                                          ; preds = %for.cond
  store i32 0, ptr addrspace(5) %i3.ascast, align 4
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc12, %for.end
  %5 = load i32, ptr addrspace(5) %i3.ascast, align 4
  %cmp5 = icmp slt i32 %5, 1000
  br i1 %cmp5, label %for.body6, label %for.end14

for.body6:                                        ; preds = %for.cond4
  %6 = load i32, ptr addrspace(5) %i3.ascast, align 4
  %rem7 = srem i32 %6, 10
  %idxprom8 = sext i32 %rem7 to i64
  %arrayidx9 = getelementptr inbounds [10 x float], ptr addrspace(5) %array.ascast, i64 0, i64 %idxprom8
; CHECK: scratch_load_b32 {{v[0-9]+}}, {{v[0-9]+}}, s33 scale_offset
  %7 = load float, ptr addrspace(5) %arrayidx9, align 4
  %8 = load i32, ptr addrspace(5) %i3.ascast, align 4
  %idxprom10 = sext i32 %8 to i64
  %arrayidx11 = getelementptr inbounds [1000 x float], ptr addrspace(10) @exchange, i64 0, i64 %idxprom10
; CHECK: scratch_store_b32 {{v[0-9]+}}, {{v[0-9]+}}, off
  store float %7, ptr addrspace(10) %arrayidx11, align 4
  br label %for.inc12

for.inc12:                                        ; preds = %for.body6
  %9 = load i32, ptr addrspace(5) %i3.ascast, align 4
  %inc13 = add nsw i32 %9, 1
  store i32 %inc13, ptr addrspace(5) %i3.ascast, align 4
  br label %for.cond4, !llvm.loop !6

for.end14:                                        ; preds = %for.cond4
  ret void
}

attributes #0 = { convergent mustprogress noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,1024" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx1300" "target-features"="+16-bit-insts,+bf16-cvt-insts,+bf16-pk-insts,+bf16-trans-insts,+bitop3-insts,+ci-insts,+dl-insts,+dot7-insts,+dot8-insts,+dpp,+fp8-conversion-insts,+gfx10-3-insts,+gfx10-insts,+gfx11-insts,+gfx12-10-insts,+gfx12-insts,+gfx13-insts,+gfx8-insts,+gfx9-insts,+permlane16-swap,+prng-inst,+tanh-insts,+wavefrontsize32" "uniform-work-group-size"="true" }

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}
