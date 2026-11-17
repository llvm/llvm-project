; Check that below 1024 x i16 mul instruction
;   %mul.ripple.LS.instance = mul <1024 x i16> %1, %0
; is lowered to HLX mpy: LX_mpy_w_h
; HLX only supports widening multiply.

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx \
; RUN: -stop-after="hexagon-isel" -verify-machineinstrs < %s | FileCheck %s

; CHECK: %[[VREG0:([0-9]+)]]:hlxlr = LX_L16b_pseudo
; CHECK: %[[VREG1:([0-9]+)]]:hlxlr = LX_L16b_pseudo
; CHECK: %[[VREG2:([0-9]+)]]:hlxllr = LX_mpy_w_h %[[VREG1]], %[[VREG0]]
; CHECK: %[[VREG3:([0-9]+)]]:hlxlr = COPY %[[VREG2]].vsub_lo
; CHECK: LX_S16b_pseudo %{{[0-9]+}}, %[[VREG3]]

target triple = "hexagon"

define dso_local void @_Z3foojPsS_S_(i32 noundef %N, ptr noalias noundef readonly %A, ptr noalias noundef readonly %B, ptr noalias noundef writeonly %C) local_unnamed_addr {
entry:
  %cmp36.not = icmp eq i32 %N, 0
  br i1 %cmp36.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:
  ret void

for.body:
  %arrayidx.phi = phi ptr [ %arrayidx.inc, %for.body ], [ %A, %entry ]
  %arrayidx3.phi = phi ptr [ %arrayidx3.inc, %for.body ], [ %B, %entry ]
  %arrayidx7.phi = phi ptr [ %arrayidx7.inc, %for.body ], [ %C, %entry ]
  %i.037 = phi i32 [ %add17, %for.body ], [ 0, %entry ]
  call void @llvm.assume(i1 true) [ "align"(ptr %arrayidx.phi, i32 2048) ]
  call void @llvm.assume(i1 true) [ "align"(ptr %arrayidx3.phi, i32 2048) ]
  call void @llvm.assume(i1 true) [ "align"(ptr %arrayidx7.phi, i32 2048) ]
  %0 = load <1024 x i16>, ptr %arrayidx.phi, align 2048
  %1 = load <1024 x i16>, ptr %arrayidx3.phi, align 2048
  %mul.ripple.LS.instance = mul <1024 x i16> %1, %0
  store <1024 x i16> %mul.ripple.LS.instance, ptr %arrayidx7.phi, align 2048
  %add17 = add i32 %i.037, 1024
  %cmp = icmp ult i32 %add17, %N
  %arrayidx.inc = getelementptr i8, ptr %arrayidx.phi, i32 2048
  %arrayidx3.inc = getelementptr i8, ptr %arrayidx3.phi, i32 2048
  %arrayidx7.inc = getelementptr i8, ptr %arrayidx7.phi, i32 2048
  br i1 %cmp, label %for.body, label %for.cond.cleanup, !llvm.loop !4
}

declare void @llvm.assume(i1 noundef)

!llvm.errno.tbaa = !{!0}

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
!4 = distinct !{!4, !5}
!5 = !{!"llvm.loop.mustprogress"}
