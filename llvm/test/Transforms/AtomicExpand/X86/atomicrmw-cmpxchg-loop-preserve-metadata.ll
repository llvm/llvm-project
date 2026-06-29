; RUN: opt -S %s -passes='require<libcall-lowering-info>,atomic-expand' -mtriple=x86_64-linux-gnu | FileCheck %s

; An atomicrmw that has no native instruction (e.g. nand) is expanded to a
; cmpxchg retry loop by insertRMWCmpXchgLoop. The pre-loop seed load
; (InitLoaded) must preserve the original !tbaa/!noalias metadata, matching the
; widened cmpxchg in the loop.

define i32 @atomicrmw_nand_preserves_metadata(ptr %p, i32 %v) {
; CHECK-LABEL: define i32 @atomicrmw_nand_preserves_metadata(
; CHECK:    load i32, ptr {{%.*}}, align 4, !tbaa ![[TBAA:[0-9]+]], !noalias ![[NOALIAS:[0-9]+]]
; CHECK:    cmpxchg ptr {{%.*}} seq_cst seq_cst, align 4, !tbaa ![[TBAA]], !noalias ![[NOALIAS]]
  %r = atomicrmw nand ptr %p, i32 %v seq_cst, align 4, !tbaa !2, !noalias !6
  ret i32 %r
}

!0 = !{!"alias-domain"}
!1 = !{!"alias-scope-a", !0}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!1}
