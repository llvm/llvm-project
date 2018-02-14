; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; FIXME: Move this to sgpr-copy.ll when this is fixed on VI.
; Make sure that when we split an smrd instruction in order to move it to
; the VALU, we are also moving its users to the VALU.

; GCN-LABEL: {{^}}split_smrd_add_worklist:
; GCN: image_sample v{{[0-9]+}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}] dmask:0x1
define amdgpu_ps void @split_smrd_add_worklist([34 x <8 x i32>] addrspace(4)* byval %arg) #0 {
bb:
  %tmp = call float @llvm.SI.load.const.v4i32(<4 x i32> undef, i32 96)
  %tmp1 = bitcast float %tmp to i32
  br i1 undef, label %bb2, label %bb3

bb2:                                              ; preds = %bb
  unreachable

bb3:                                              ; preds = %bb
  %tmp4 = bitcast float %tmp to i32
  %tmp5 = add i32 %tmp4, 4
  %tmp6 = sext i32 %tmp5 to i64
  %tmp7 = getelementptr [34 x <8 x i32>], [34 x <8 x i32>] addrspace(4)* %arg, i64 0, i64 %tmp6
  %tmp8 = load <8 x i32>, <8 x i32> addrspace(4)* %tmp7, align 32, !tbaa !0
  %tmp9 = call <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float> <float bitcast (i32 1061158912 to float), float bitcast (i32 1048576000 to float)>, <8 x i32> %tmp8, <4 x i32> undef, i32 15, i1 false, i1 false, i1 false, i1 false, i1 false)
  %tmp10 = extractelement <4 x float> %tmp9, i32 0
  %tmp12 = call <2 x half> @llvm.amdgcn.cvt.pkrtz(float %tmp10, float undef)
  call void @llvm.amdgcn.exp.compr.v2f16(i32 0, i32 15, <2 x half> %tmp12, <2 x half> undef, i1 true, i1 true) #0
  ret void
}

declare <2 x half> @llvm.amdgcn.cvt.pkrtz(float, float) #1
declare void @llvm.amdgcn.exp.compr.v2f16(i32, i32, <2 x half>, <2 x half>, i1, i1) #0
declare <4 x float> @llvm.amdgcn.image.sample.v4f32.v2f32.v8i32(<2 x float>, <8 x i32>, <4 x i32>, i32, i1, i1, i1, i1, i1) #2
declare float @llvm.SI.load.const.v4i32(<4 x i32>, i32) #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind readonly }

!0 = !{!1, !1, i64 0, i32 1}
!1 = !{!"const", !2}
!2 = !{!"tbaa root"}
