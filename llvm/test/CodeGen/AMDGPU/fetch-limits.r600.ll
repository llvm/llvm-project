; RUN: llc < %s -march=r600 -mcpu=r600 | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=rs880 | FileCheck %s
; RUN: llc < %s -march=r600 -mcpu=rv670 | FileCheck %s

; R600 supports 8 fetches in a clause
; CHECK: {{^}}fetch_limits_r600:
; CHECK: Fetch clause
; CHECK: Fetch clause

define amdgpu_ps void @fetch_limits_r600() {
entry:
  %tmp = load <4 x float>, ptr addrspace(8) null
  %tmp1 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 1)
  %tmp2 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 2)
  %tmp3 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 3)
  %tmp4 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 4)
  %tmp5 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 5)
  %tmp6 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 6)
  %tmp7 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 7)
  %tmp8 = load <4 x float>, ptr addrspace(8) getelementptr ([1024 x <4 x float>], ptr addrspace(8) null, i64 0, i32 8)
  %tmp9 = shufflevector <4 x float> %tmp, <4 x float> %tmp, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp10 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp9, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp11 = shufflevector <4 x float> %tmp1, <4 x float> %tmp1, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp12 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp11, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp13 = shufflevector <4 x float> %tmp2, <4 x float> %tmp2, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp14 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp13, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp15 = shufflevector <4 x float> %tmp3, <4 x float> %tmp3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp16 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp17 = shufflevector <4 x float> %tmp4, <4 x float> %tmp4, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp18 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp17, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp19 = shufflevector <4 x float> %tmp5, <4 x float> %tmp5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp20 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp19, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp21 = shufflevector <4 x float> %tmp6, <4 x float> %tmp6, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp22 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp21, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp23 = shufflevector <4 x float> %tmp7, <4 x float> %tmp7, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp24 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp23, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %tmp25 = shufflevector <4 x float> %tmp8, <4 x float> %tmp8, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %tmp26 = call <4 x float> @llvm.r600.tex(<4 x float> %tmp25, i32 0, i32 0, i32 0, i32 0, i32 0, i32 1, i32 1, i32 1, i32 1)
  %a = fadd <4 x float> %tmp10, %tmp12
  %b = fadd <4 x float> %tmp14, %tmp16
  %c = fadd <4 x float> %tmp18, %tmp20
  %d = fadd <4 x float> %tmp22, %tmp24
  %e = fadd <4 x float> %tmp26, %a
  %bc = fadd <4 x float> %b, %c
  %de = fadd <4 x float> %d, %e
  %bcde = fadd <4 x float> %bc, %de
  call void @llvm.r600.store.swizzle(<4 x float> %bcde, i32 0, i32 1)
  ret void
}

declare void @llvm.r600.store.swizzle(<4 x float>, i32, i32)

; Function Attrs: readnone
declare <4 x float> @llvm.r600.tex(<4 x float>, i32, i32, i32, i32, i32, i32, i32, i32, i32) #0

attributes #0 = { nounwind readnone }
