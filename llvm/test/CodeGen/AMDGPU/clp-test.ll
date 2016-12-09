; RUN: opt -amdgpu-clp-vector-expansion -mtriple=amdgcn-- -mcpu=fiji -S < %s | FileCheck %s
; Function Attrs: nounwind
define void @foo(float %p1, <2 x float> %p2, <3 x float> %p3, <4 x float> %p4) {
entry:
  call <2 x float> @_Z4cbrtDv2_f(<2 x float> %p2)
  call <3 x float> @_Z3madDv3_fS_S_(<3 x float> %p3, <3 x float> %p3, <3 x float> %p3)
  call <4 x float> @_Z4fmaxDv4_ff(<4 x float> %p4, float %p1)
  ret void
}

; CHECK: define weak <2 x float> @_Z4cbrtDv2_f(<2 x float> %_p1)
; CHECK: %[[var0:[0-9]+]] = extractelement <2 x float> %_p1, i32 0
; CHECK: %[[var1:[0-9]+]] = extractelement <2 x float> %_p1, i32 1
; CHECK: %lo.call = call float @_Z4cbrtf(float %[[var0]])
; CHECK: %[[var2:[0-9]+]] = insertelement <2 x float> undef, float %lo.call, i32 0
; CHECK: %hi.call = call float @_Z4cbrtf(float %[[var1]])
; CHECK: %[[var3:[0-9]+]] = insertelement <2 x float> %[[var2]], float %hi.call, i32 1
; CHECK: ret <2 x float> %[[var3]]

; Function Attrs: nounwind readnone
declare <2 x float> @_Z4cbrtDv2_f(<2 x float>)
; CHECK: declare float @_Z4cbrtf(float)
; CHECK-NOT: declare <2 x float> @_Z4cbrtDv2_f(<2 x float>)

; CHECK: define weak <3 x float> @_Z3madDv3_fS_S_(<3 x float> %_p1, <3 x float> %_p2, <3 x float> %_p3)
; CHECK: %lo.call = call <2 x float> @_Z3madDv2_fS_S_(<2 x float> %{{[0-9]+}}, <2 x float> %{{[0-9]+}}, <2 x float> %{{[0-9]+}})
; CHECK: %[[var4:[0-9]+]] = extractelement <2 x float> %lo.call, i32 0
; CHECK: %[[var5:[0-9]+]] = insertelement <3 x float> undef, float %[[var4]], i32 0
; CHECK: %[[var6:[0-9]+]] = extractelement <2 x float> %lo.call, i32 1
; CHECK: %[[var7:[0-9]+]] = insertelement <3 x float> %[[var5]], float %[[var6]], i32 1
; CHECK: %hi.call = call float @_Z3madfff(float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}})
; CHECK: %[[var8:[0-9]+]] = insertelement <3 x float> %[[var7]], float %hi.call, i32 2
; CHECK: ret <3 x float> %[[var8]]

; CHECK: define weak <2 x float> @_Z3madDv2_fS_S_(<2 x float> %_p1, <2 x float> %_p2, <2 x float> %_p3)
; CHECK: %lo.call = call float @_Z3madfff(float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}})
; CHECK: %[[var9:[0-9]+]] = insertelement <2 x float> undef, float %lo.call, i32 0
; CHECK: %hi.call = call float @_Z3madfff(float %{{[0-9]+}}, float %{{[0-9]+}}, float %{{[0-9]+}})
; CHECK: %[[var10:[0-9]+]] = insertelement <2 x float> %[[var9]], float %hi.call, i32 1
; CHECK: ret <2 x float> %[[var10:[0-9]+]]

declare <3 x float> @_Z3madDv3_fS_S_(<3 x float>, <3 x float>, <3 x float>)
; CHECK: declare float @_Z3madfff(float, float, float)
; CHECK-NOT: declare <3 x float> @_Z3madDv3_fS_S_(<3 x float>, <3 x float>, <3 x float>)

; CHECK: define weak <4 x float> @_Z4fmaxDv4_ff(<4 x float> %_p1, float %_p2)
; CHECK: %hi.call = call <2 x float> @_Z4fmaxDv2_ff(<2 x float> %{{[0-9]+}}, float %_p2)

; CHECK: define weak <2 x float> @_Z4fmaxDv2_ff(<2 x float> %_p1, float %_p2)
; CHECK: %lo.call = call float @_Z4fmaxff(float %{{[0-9]+}}, float %_p2)
; CHECK: %[[var11:[0-9]+]] = insertelement <2 x float> undef, float %lo.call, i32 0
; CHECK: %hi.call = call float @_Z4fmaxff(float %{{[0-9]+}}, float %_p2)
; CHECK: %[[var12:[0-9]+]] = insertelement <2 x float> %[[var11]], float %hi.call, i32 1
; CHECK: ret <2 x float> %[[var12]]

declare <4 x float> @_Z4fmaxDv4_ff(<4 x float>, float)
; CHECK: declare float @_Z4fmaxff(float, float)
; CHECK-NOT: declare <2 x float> @_Z4fmaxDv2_ff(<2 x float>, float)
