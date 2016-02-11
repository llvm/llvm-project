; RUN: opt -S -codegenprepare -mtriple=amdgcn-unknown-unknown < %s | FileCheck -check-prefix=ASC -check-prefix=COMMON %s

; COMMON-LABEL: @test_sink_ptrtoint_asc(
; ASC: addrspacecast
; ASC-NOT: ptrtoint
; ASC-NOT: inttoptr

define void @test_sink_ptrtoint_asc(float addrspace(1)* nocapture %arg, float addrspace(1)* nocapture readonly %arg1, float addrspace(3)* %arg2) #0 {
bb:
  %tmp = getelementptr inbounds float, float addrspace(3)* %arg2, i32 16
  %tmp2 = tail call i32 @llvm.amdgcn.workitem.id.x() #1
  %tmp3 = sext i32 %tmp2 to i64
  %tmp4 = getelementptr inbounds float, float addrspace(1)* %arg1, i64 %tmp3
  %tmp5 = load float, float addrspace(1)* %tmp4, align 4
  %tmp6 = addrspacecast float addrspace(3)* %tmp to float addrspace(4)*
  %tmp7 = fcmp olt float %tmp5, 8.388608e+06
  br i1 %tmp7, label %bb8, label %bb14

bb8:                                              ; preds = %bb
  %tmp9 = tail call float @llvm.fma.f32(float %tmp5, float 0x3FE45F3060000000, float 5.000000e-01) #1
  %tmp10 = fmul float %tmp9, 0x3E74442D00000000
  %tmp11 = fsub float -0.000000e+00, %tmp10
  %tmp12 = tail call float @llvm.fma.f32(float %tmp9, float 0x3E74442D00000000, float %tmp11) #1
  store float %tmp12, float addrspace(4)* %tmp6, align 4
  %tmp13 = fsub float -0.000000e+00, %tmp12
  br label %bb15

bb14:                                             ; preds = %bb
  store float 2.000000e+00, float addrspace(4)* %tmp6, align 4
  br label %bb15

bb15:                                             ; preds = %bb14, %bb8
  %tmp16 = phi float [ 0.000000e+00, %bb14 ], [ %tmp13, %bb8 ]
  %tmp17 = fsub float -0.000000e+00, %tmp16
  %tmp18 = tail call float @llvm.fma.f32(float 1.000000e+00, float 0x3FF0AAAAA0000000, float %tmp17) #1
  %tmp19 = fsub float 2.187500e-01, %tmp18
  %tmp20 = fsub float 7.187500e-01, %tmp19
  %tmp21 = fcmp ogt float %tmp5, 1.600000e+01
  %tmp22 = select i1 %tmp21, float 0x7FF8000000000000, float %tmp20
  %tmp23 = getelementptr inbounds float, float addrspace(1)* %arg, i64 %tmp3
  store float %tmp22, float addrspace(1)* %tmp23, align 4
  ret void
}

declare float @llvm.fma.f32(float, float, float) #1
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
