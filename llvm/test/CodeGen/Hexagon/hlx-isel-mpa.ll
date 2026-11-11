; Check that below 1024 x i16 mul and add instructions
;  %mul.ripple.vectorized = mul <1024 x i16> %0, %.ripple.LS.instance.ripple.bcast.splat
;  %add25.ripple.LS.instance = add <1024 x i16> %mul.ripple.vectorized, %acc.068.ripple.vectorized
; are lowered to HLX mpy-add: LX_mpa_wh
; HLX only supports widening multiply.

; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx \
; RUN: -stop-after="hexagon-isel" -verify-machineinstrs < %s | FileCheck %s

; CHECK: bb.{{[0-9]+}}.for.body8:
; CHECK: %[[VREG0:([0-9]+)]]:hlxlr = PHI %{{[0-9]+}}, %bb.{{[0-9]+}}, %[[VREG1:([0-9]+)]], %bb.{{[0-9]+}}
; CHECK: %[[REG0:([0-9]+)]]:intregs = L2_loadruh_io
; CHECK: %[[VREG2:([0-9]+)]]:hlxlr = LX_assign_r %[[REG0]]
; CHECK: %[[VREG3:([0-9]+)]]:hlxlr = LX_L16b_pseudo
; CHECK: %[[VREG4:([0-9]+)]]:hlxlr = LX_assign_s 0
; CHECK: %[[VREG5:([0-9]+)]]:hlxllr = LX_combine_w %[[VREG4]], %[[VREG0]]
; CHECK: %[[VREG6:([0-9]+)]]:hlxllr = LX_mpa_wh %[[VREG5]], %[[VREG3]], %[[VREG2]]
; CHECK: %[[VREG1]]:hlxlr = COPY %[[VREG6]].vsub_lo

target triple = "hexagon"

$_Z14AT_dot_B_basicILj1024ELj1024ELj1ELj1024EEvjPAT__KsPAT0__S0_PAT0__s = comdat any

define linkonce_odr dso_local void @_Z14AT_dot_B_basicILj1024ELj1024ELj1ELj1024EEvjPAT__KsPAT0__S0_PAT0__s(i32 noundef %Nk, ptr noalias noundef %A, ptr noalias noundef %B, ptr noalias noundef %C) local_unnamed_addr comdat {
entry:
  %cmp666.not = icmp eq i32 %Nk, 0
  br label %for.cond1.preheader

for.cond1.preheader:
  %arrayidx28.phi.ripple.LS.instance = phi ptr [ %C, %entry ], [ %arrayidx28.inc, %for.cond.cleanup7 ]
  %i.071 = phi i32 [ 0, %entry ], [ %add39, %for.cond.cleanup7 ]
  br i1 %cmp666.not, label %for.cond.cleanup7, label %for.body8.preheader

for.body8.preheader:
  %invariant.gep = getelementptr i16, ptr %A, i32 %i.071
  br label %for.body8

for.cond.cleanup:
  ret void

for.cond.cleanup7:
  %acc.0.lcssa.ripple.LS.instance = phi <1024 x i16> [ zeroinitializer, %for.cond1.preheader ], [ %add25.ripple.LS.instance, %for.body8 ]
  call void @llvm.assume(i1 true) [ "align"(ptr %arrayidx28.phi.ripple.LS.instance, i32 2048) ]
  store <1024 x i16> %acc.0.lcssa.ripple.LS.instance, ptr %arrayidx28.phi.ripple.LS.instance, align 2048
  %add39 = add nuw nsw i32 %i.071, 1
  %exitcond72.not = icmp eq i32 %add39, 1024
  %arrayidx28.inc = getelementptr i8, ptr %arrayidx28.phi.ripple.LS.instance, i32 2048
  br i1 %exitcond72.not, label %for.cond.cleanup, label %for.cond1.preheader, !llvm.loop !7

for.body8:
  %acc.068.ripple.vectorized = phi <1024 x i16> [ %add25.ripple.LS.instance, %for.body8 ], [ zeroinitializer, %for.body8.preheader ]
  %gep.phi.ripple.LS.instance = phi ptr [ %gep.inc, %for.body8 ], [ %invariant.gep, %for.body8.preheader ]
  %arrayidx12.phi.ripple.LS.instance = phi ptr [ %arrayidx12.inc, %for.body8 ], [ %B, %for.body8.preheader ]
  %k.067.apmt = phi i32 [ %inc.apmt, %for.body8 ], [ 0, %for.body8.preheader ]
  call void @llvm.assume(i1 true) [ "align"(ptr %gep.phi.ripple.LS.instance, i32 2048) ]
  call void @llvm.assume(i1 true) [ "align"(ptr %arrayidx12.phi.ripple.LS.instance, i32 2048) ]
  %.ripple.LS.instance = load i16, ptr %gep.phi.ripple.LS.instance, align 2048, !tbaa !4
  %.ripple.LS.instance.ripple.bcast.splatinsert = insertelement <1024 x i16> poison, i16 %.ripple.LS.instance, i64 0
  %.ripple.LS.instance.ripple.bcast.splat = shufflevector <1024 x i16> %.ripple.LS.instance.ripple.bcast.splatinsert, <1024 x i16> poison, <1024 x i32> zeroinitializer
  %0 = load <1024 x i16>, ptr %arrayidx12.phi.ripple.LS.instance, align 2048
  %mul.ripple.vectorized = mul <1024 x i16> %0, %.ripple.LS.instance.ripple.bcast.splat
  %add25.ripple.LS.instance = add <1024 x i16> %mul.ripple.vectorized, %acc.068.ripple.vectorized
  %inc.apmt = add nuw i32 %k.067.apmt, 1
  %exitcond.not = icmp eq i32 %inc.apmt, %Nk
  %gep.inc = getelementptr i8, ptr %gep.phi.ripple.LS.instance, i32 2048
  %arrayidx12.inc = getelementptr i8, ptr %arrayidx12.phi.ripple.LS.instance, i32 2048
  br i1 %exitcond.not, label %for.cond.cleanup7, label %for.body8, !llvm.loop !8
}

declare void @llvm.assume(i1 noundef)

!llvm.errno.tbaa = !{!0}

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"short", !2, i64 0}
!6 = !{!"llvm.loop.mustprogress"}
!7 = distinct !{!7, !6}
!8 = distinct !{!8, !6}
