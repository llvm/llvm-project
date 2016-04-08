; RUN: opt -S -instcombine < %s | FileCheck %s

define float @test1(i32 %scale) {
entry:
  %tmp1 = icmp sgt i32 1, %scale
  %tmp2 = select i1 %tmp1, i32 1, i32 %scale
  %tmp3 = sitofp i32 %tmp2 to float
  %tmp4 = icmp sgt i32 %tmp2, 0
  %sel = select i1 %tmp4, float %tmp3, float 0.000000e+00
  ret float %sel
}

; CHECK-LABEL: define float @test1(
; CHECK:  %[[tmp1:.*]] = icmp slt i32 %scale, 1
; CHECK:  %[[tmp2:.*]] = select i1 %[[tmp1]], i32 1, i32 %scale
; CHECK:  %[[tmp3:.*]] = sitofp i32 %[[tmp2]] to float
; CHECK:  %[[tmp4:.*]] = icmp sgt i32 %[[tmp2]], 0
; CHECK:  %[[sel:.*]] = select i1 %[[tmp4]], float %[[tmp3]], float 0.000000e+00
; CHECK:  ret float %[[sel]]
