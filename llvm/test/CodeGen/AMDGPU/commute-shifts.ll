; RUN: llc -march=amdgcn -mcpu=verde -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=GCN -check-prefix=VI %s

; GCN-LABEL: {{^}}main:
; SI: v_lshl_b32_e32 v{{[0-9]+}}, 1, v{{[0-9]+}}
; VI: v_lshlrev_b32_e64 v{{[0-9]+}}, v{{[0-9]+}}, 1
define amdgpu_ps void @main(float %arg0, float %arg1) #0 {
bb:
  %tmp = fptosi float %arg0 to i32
  %tmp1 = call <4 x float> @llvm.SI.image.load.v4i32(<4 x i32> undef, <8 x i32> undef, i32 15, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0, i32 0)
  %tmp2.f = extractelement <4 x float> %tmp1, i32 0
  %tmp2 = bitcast float %tmp2.f to i32
  %tmp3 = and i32 %tmp, 7
  %tmp4 = shl i32 1, %tmp3
  %tmp5 = and i32 %tmp2, %tmp4
  %tmp6 = icmp eq i32 %tmp5, 0
  %tmp7 = select i1 %tmp6, float 0.000000e+00, float %arg1
  %tmp8 = call i32 @llvm.SI.packf16(float undef, float %tmp7)
  %tmp9 = bitcast i32 %tmp8 to float
  call void @llvm.SI.export(i32 15, i32 1, i32 1, i32 0, i32 1, float undef, float %tmp9, float undef, float %tmp9)
  ret void
}

declare <4 x float> @llvm.SI.image.load.v4i32(<4 x i32>, <8 x i32>, i32, i32, i32, i32, i32, i32, i32, i32) #1
declare i32 @llvm.SI.packf16(float, float) #1
declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
