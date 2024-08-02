; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -amdgpu-promote-lane-shared=false -stop-after=finalize-isel -verify-machineinstrs -o - %s | FileCheck %s
; RUN: llc -mtriple=amdgcn-- -mcpu=gfx1300 -verify-machineinstrs -stop-after=finalize-isel -o - %s | FileCheck -check-prefix=VIDX %s
target datalayout = "A5"

@exchange = external addrspace(10) global [70 x float], align 4

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define amdgpu_kernel void @_Z3foov() noinline optnone {
entry:
  %array.ascast = alloca [10 x float], align 4, addrspace(5)
  %i.ascast = alloca i32, align 4, addrspace(5)
  %i3.ascast = alloca i32, align 4, addrspace(5)
  store i32 0, ptr addrspace(5) %i.ascast, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr addrspace(5) %i.ascast, align 4
  %cmp = icmp slt i32 %0, 70
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr addrspace(5) %i.ascast, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [70 x float], ptr addrspace(10) @exchange, i64 17, i64 %idxprom
; CHECK: [[INLS:%[0-9]+]]:vgpr_32 = SCRATCH_LOAD_DWORD killed {{%[0-9]+}}, 4760, 0, implicit $exec, implicit $flat_scr :: (load (s32) from %ir.arrayidx, addrspace 10)
; VIDX: [[RFL0:%[0-9]+]]:sgpr_32 = V_READFIRSTLANE_B32
; VIDX-NEXT: [[CMP0:%[0-9]+]]:sreg_32 = V_CMP_EQ_U32_e64 [[RFL0]], {{%[0-9]+}}, implicit $exec
; VIDX-NEXT: [[ANDEXEC0:%[0-9]+]]:sreg_32 = S_AND_SAVEEXEC_B32 killed [[CMP0]], implicit-def $exec, implicit-def $scc, implicit $exec
; VIDX-NEXT: [[DIV4A:%[0-9]+]]:sgpr_32 = S_LSHR_B32 [[RFL0]], 2, implicit-def dead $scc
; VIDX-NEXT: [[ADD0:%[0-9]+]]:sgpr_32 = S_ADD_I32 [[DIV4A]], 1190, implicit-def dead $scc
; VIDX-NEXT: [[LOAD:%[0-9]+]]:vgpr_32 = V_LOAD_IDX [[ADD0]], 0, 1, implicit $exec
; VIDX-NEXT: $exec_lo = S_XOR_B32_term $exec_lo, [[ANDEXEC0]], implicit-def $scc
  %2 = load float, ptr addrspace(10) %arrayidx, align 4
  %3 = load i32, ptr addrspace(5) %i.ascast, align 4
  %rem = srem i32 %3, 10
  %idxprom1 = sext i32 %rem to i64
  %arrayidx2 = getelementptr inbounds [10 x float], ptr addrspace(5) %array.ascast, i64 0, i64 %idxprom1
; CHECK: SCRATCH_STORE_DWORD_SVS killed [[INLS]], killed {{%[0-9]+}}, %stack.0.array.ascast, 0, 2048, implicit $exec, implicit $flat_scr :: (store (s32) into %ir.arrayidx2, addrspace 5)
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
  %cmp5 = icmp slt i32 %5, 70
  br i1 %cmp5, label %for.body6, label %for.end14

for.body6:                                        ; preds = %for.cond4
  %6 = load i32, ptr addrspace(5) %i3.ascast, align 4
  %rem7 = srem i32 %6, 10
  %idxprom8 = sext i32 %rem7 to i64
  %arrayidx9 = getelementptr inbounds [10 x float], ptr addrspace(5) %array.ascast, i64 0, i64 %idxprom8
; CHECK: [[TOLS:%[0-9]+]]:vgpr_32 = SCRATCH_LOAD_DWORD_SVS killed {{%[0-9]+}}, %stack.0.array.ascast, 0, 2048, implicit $exec, implicit $flat_scr :: (load (s32) from %ir.arrayidx9, addrspace 5)
  %7 = load float, ptr addrspace(5) %arrayidx9, align 4
  %8 = load i32, ptr addrspace(5) %i3.ascast, align 4
  %idxprom10 = sext i32 %8 to i64
  %arrayidx11 = getelementptr inbounds [70 x float], ptr addrspace(10) @exchange, i64 5, i64 %idxprom10
; CHECK: SCRATCH_STORE_DWORD killed [[TOLS]], killed {{%[0-9]+}}, 1400, 0, implicit $exec, implicit $flat_scr :: (store (s32) into %ir.arrayidx11, addrspace 10)
; VIDX: [[RFL1:%[0-9]+]]:sgpr_32 = V_READFIRSTLANE_B32
; VIDX-NEXT: [[CMP1:%[0-9]+]]:sreg_32 = V_CMP_EQ_U32_e64 [[RFL1]], {{%[0-9]+}}, implicit $exec
; VIDX-NEXT: [[ANDEXEC1:%[0-9]+]]:sreg_32 = S_AND_SAVEEXEC_B32 killed [[CMP1]], implicit-def $exec, implicit-def $scc, implicit $exec
; VIDX-NEXT: [[DIV4B:%[0-9]+]]:sgpr_32 = S_LSHR_B32 [[RFL1]], 2, implicit-def dead $scc
; VIDX-NEXT: V_STORE_IDX {{%[0-9]+}}, [[DIV4B]], 350, 1, implicit $exec
; VIDX-NEXT: $exec_lo = S_XOR_B32_term $exec_lo, [[ANDEXEC1]], implicit-def $scc
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

!llvm.module.flags = !{!0, !1, !2}

!0 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
!6 = distinct !{!6, !5}
