; RUN: llc %s -mtriple=aarch64-eabi -stop-after=finalize-isel -o - | FileCheck --check-prefix=MIR %s

; Ensure the scoped AA metadata is still retained after store merging.

; MIR-DAG: ![[DOMAIN:[0-9]+]] = distinct !{!{{[0-9]+}}, !"domain"}
; MIR-DAG: ![[SCOPE0:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"scope 0"}
; MIR-DAG: ![[SCOPE1:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"scope 1"}
; MIR-DAG: ![[SCOPE2:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"scope 2"}
; MIR-DAG: ![[SCOPE3:[0-9]+]] = distinct !{!{{[0-9]+}}, ![[DOMAIN]], !"scope 3"}
; MIR-DAG: ![[SET0:[0-9]+]] = !{![[SCOPE0]]}
; MIR-DAG: ![[SET1:[0-9]+]] = !{![[SCOPE1]]}
; MIR-DAG: ![[SET2:[0-9]+]] = !{![[SCOPE2]]}
; MIR-DAG: ![[SET3:[0-9]+]] = !{![[SCOPE3]]}

define void @blam0(ptr %g0, ptr %g1) {
; MIR-LABEL: name: blam0
; MIR: LDRDui %0, 0 :: (load (s64) from %ir.g0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR: STRDui killed %5, %1, 0 :: (store (s64) into %ir.tmp41, align 4, !alias.scope ![[SET1]], !noalias ![[SET0]])
  %tmp4 = getelementptr inbounds <3 x float>, ptr %g1, i64 0, i64 0
  %tmp5 = load <3 x float>, ptr %g0, align 4, !alias.scope !0, !noalias !1
  %tmp6 = extractelement <3 x float> %tmp5, i64 0
  store float %tmp6, ptr %tmp4, align 4, !alias.scope !1, !noalias !0
  %tmp7 = getelementptr inbounds float, ptr %tmp4, i64 1
  %tmp8 = load <3 x float>, ptr %g0, align 4, !alias.scope !0, !noalias !1
  %tmp9 = extractelement <3 x float> %tmp8, i64 1
  store float %tmp9, ptr %tmp7, align 4, !alias.scope !1, !noalias !0
  ret void;
}

; Ensure new scoped AA metadata are calculated after merging stores.
define void @blam1(ptr %g0, ptr %g1) {
; MIR-LABEL: name: blam1
; MIR: machineMetadataNodes:
; MIR-DAG: ![[MMSET0:[0-9]+]] = !{![[SCOPE2]], ![[SCOPE1]]}
; MIR-DAG: ![[MMSET1:[0-9]+]] = !{}
; MIR: body:
; MIR: LDRDui %0, 0 :: (load (s64) from %ir.g0, align 4, !alias.scope ![[SET0]], !noalias ![[SET1]])
; MIR: STRDui killed %5, %1, 0 :: (store (s64) into %ir.tmp41, align 4, !alias.scope ![[MMSET0]], !noalias ![[MMSET1]])
  %tmp4 = getelementptr inbounds <3 x float>, ptr %g1, i64 0, i64 0
  %tmp5 = load <3 x float>, ptr %g0, align 4, !alias.scope !0, !noalias !1
  %tmp6 = extractelement <3 x float> %tmp5, i64 0
  store float %tmp6, ptr %tmp4, align 4, !alias.scope !1, !noalias !0
  %tmp7 = getelementptr inbounds float, ptr %tmp4, i64 1
  %tmp8 = load <3 x float>, ptr %g0, align 4, !alias.scope !0, !noalias !1
  %tmp9 = extractelement <3 x float> %tmp8, i64 1
  store float %tmp9, ptr %tmp7, align 4, !alias.scope !5, !noalias !6
  ret void;
}

!0 = !{!2}
!1 = !{!3}
!2 = !{!2, !4, !"scope 0"}
!3 = !{!3, !4, !"scope 1"}
!4 = !{!4, !"domain"}
!5 = !{!7}
!6 = !{!8}
!7 = !{!7, !4, !"scope 2"}
!8 = !{!8, !4, !"scope 3"}
