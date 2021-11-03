; REQUIRES: asserts

; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs -debug-only=machine-scheduler -o /dev/null < %s 2>&1 | FileCheck %s

; We are only targeting one wave. Check that the machine scheduler doesn't use
; register pressure heuristics to prioritize any candidate instruction.

; CHECK-NOT: REG-CRIT
; CHECK-NOT: REG-EXCESS

define amdgpu_kernel void @load_fma_store(float addrspace(3)* nocapture readonly %arg, float addrspace(3)* nocapture %arg1) #1 {
bb:
  %tmp0 = getelementptr inbounds float, float addrspace(3)* %arg, i32 1
  %tmp1 = load float, float addrspace(3)* %tmp0, align 4
  %tmp2 = getelementptr inbounds float, float addrspace(3)* %arg, i32 2
  %tmp3 = load float, float addrspace(3)* %tmp2, align 4
  %tmp4 = getelementptr inbounds float, float addrspace(3)* %arg, i32 3
  %tmp5 = load float, float addrspace(3)* %tmp4, align 4
  %tmp6 = getelementptr inbounds float, float addrspace(3)* %arg, i32 4
  %tmp7 = load float, float addrspace(3)* %tmp6, align 4
  %tmp8 = getelementptr inbounds float, float addrspace(3)* %arg, i32 5
  %tmp9 = load float, float addrspace(3)* %tmp8, align 4
  %tmp10 = getelementptr inbounds float, float addrspace(3)* %arg, i32 6
  %tmp11 = load float, float addrspace(3)* %tmp10, align 4
  %tmp12 = getelementptr inbounds float, float addrspace(3)* %arg, i32 7
  %tmp13 = load float, float addrspace(3)* %tmp12, align 4
  %tmp14 = getelementptr inbounds float, float addrspace(3)* %arg, i32 8
  %tmp15 = load float, float addrspace(3)* %tmp14, align 4
  %tmp16 = getelementptr inbounds float, float addrspace(3)* %arg, i32 9
  %tmp17 = load float, float addrspace(3)* %tmp16, align 4
  %tmp18 = getelementptr inbounds float, float addrspace(3)* %arg, i32 10
  %tmp19 = load float, float addrspace(3)* %tmp18, align 4
  %tmp20 = getelementptr inbounds float, float addrspace(3)* %arg, i32 11
  %tmp21 = load float, float addrspace(3)* %tmp20, align 4
  %tmp22 = getelementptr inbounds float, float addrspace(3)* %arg, i32 12
  %tmp23 = load float, float addrspace(3)* %tmp22, align 4
  %tmp24 = getelementptr inbounds float, float addrspace(3)* %arg, i32 13
  %tmp25 = load float, float addrspace(3)* %tmp24, align 4
  %tmp26 = getelementptr inbounds float, float addrspace(3)* %arg, i32 14
  %tmp27 = load float, float addrspace(3)* %tmp26, align 4
  %tmp28 = getelementptr inbounds float, float addrspace(3)* %arg, i32 15
  %tmp29 = load float, float addrspace(3)* %tmp28, align 4
  %tmp30 = getelementptr inbounds float, float addrspace(3)* %arg, i32 16
  %tmp31 = load float, float addrspace(3)* %tmp30, align 4
  %tmp32 = getelementptr inbounds float, float addrspace(3)* %arg, i32 17
  %tmp33 = load float, float addrspace(3)* %tmp32, align 4
  %tmp34 = getelementptr inbounds float, float addrspace(3)* %arg, i32 18
  %tmp35 = load float, float addrspace(3)* %tmp34, align 4
  %tmp36 = getelementptr inbounds float, float addrspace(3)* %arg, i32 19
  %tmp37 = load float, float addrspace(3)* %tmp36, align 4
  %tmp38 = getelementptr inbounds float, float addrspace(3)* %arg, i32 20
  %tmp39 = load float, float addrspace(3)* %tmp38, align 4
  %tmp40 = getelementptr inbounds float, float addrspace(3)* %arg, i32 21
  %tmp41 = load float, float addrspace(3)* %tmp40, align 4
  %tmp42 = getelementptr inbounds float, float addrspace(3)* %arg, i32 22
  %tmp43 = load float, float addrspace(3)* %tmp42, align 4
  %tmp44 = getelementptr inbounds float, float addrspace(3)* %arg, i32 23
  %tmp45 = load float, float addrspace(3)* %tmp44, align 4
  %tmp46 = getelementptr inbounds float, float addrspace(3)* %arg, i32 24
  %tmp47 = load float, float addrspace(3)* %tmp46, align 4
  %tmp48 = getelementptr inbounds float, float addrspace(3)* %arg, i32 25
  %tmp49 = load float, float addrspace(3)* %tmp48, align 4
  %tmp50 = getelementptr inbounds float, float addrspace(3)* %arg, i32 26
  %tmp51 = load float, float addrspace(3)* %tmp50, align 4
  %tmp52 = getelementptr inbounds float, float addrspace(3)* %arg, i32 27
  %tmp53 = load float, float addrspace(3)* %tmp52, align 4
  %tmp54 = getelementptr inbounds float, float addrspace(3)* %arg, i32 28
  %tmp55 = load float, float addrspace(3)* %tmp54, align 4
  %tmp56 = getelementptr inbounds float, float addrspace(3)* %arg, i32 29
  %tmp57 = load float, float addrspace(3)* %tmp56, align 4
  %tmp58 = getelementptr inbounds float, float addrspace(3)* %arg, i32 30
  %tmp59 = load float, float addrspace(3)* %tmp58, align 4
  %tmp60 = tail call float @llvm.fmuladd.f32(float %tmp1, float %tmp3, float %tmp5)
  %tmp61 = tail call float @llvm.fmuladd.f32(float %tmp7, float %tmp9, float %tmp11)
  %tmp62 = tail call float @llvm.fmuladd.f32(float %tmp13, float %tmp15, float %tmp17)
  %tmp63 = tail call float @llvm.fmuladd.f32(float %tmp19, float %tmp21, float %tmp23)
  %tmp64 = tail call float @llvm.fmuladd.f32(float %tmp25, float %tmp27, float %tmp29)
  %tmp65 = tail call float @llvm.fmuladd.f32(float %tmp31, float %tmp33, float %tmp35)
  %tmp66 = tail call float @llvm.fmuladd.f32(float %tmp37, float %tmp39, float %tmp41)
  %tmp67 = tail call float @llvm.fmuladd.f32(float %tmp43, float %tmp45, float %tmp47)
  %tmp68 = tail call float @llvm.fmuladd.f32(float %tmp49, float %tmp51, float %tmp53)
  %tmp69 = tail call float @llvm.fmuladd.f32(float %tmp55, float %tmp57, float %tmp59)
  %tmp70 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 1
  store float %tmp60, float addrspace(3)* %tmp70, align 4
  %tmp71 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 2
  store float %tmp61, float addrspace(3)* %tmp71, align 4
  %tmp72 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 3
  store float %tmp62, float addrspace(3)* %tmp72, align 4
  %tmp73 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 4
  store float %tmp63, float addrspace(3)* %tmp73, align 4
  %tmp74 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 5
  store float %tmp64, float addrspace(3)* %tmp74, align 4
  %tmp75 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 6
  store float %tmp65, float addrspace(3)* %tmp75, align 4
  %tmp76 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 7
  store float %tmp66, float addrspace(3)* %tmp76, align 4
  %tmp77 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 8
  store float %tmp67, float addrspace(3)* %tmp77, align 4
  %tmp78 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 9
  store float %tmp68, float addrspace(3)* %tmp78, align 4
  %tmp79 = getelementptr inbounds float, float addrspace(3)* %arg1, i64 10
  store float %tmp69, float addrspace(3)* %tmp79, align 4
  ret void
}

; Function Attrs: nounwind readnone
declare float @llvm.fmuladd.f32(float, float, float) #0

attributes #0 = { nounwind readnone }
attributes #1 = { "amdgpu-waves-per-eu"="1,1" "amdgpu-flat-work-group-size"="1,256" }
