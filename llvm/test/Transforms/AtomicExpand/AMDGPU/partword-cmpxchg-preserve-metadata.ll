; RUN: opt -mtriple=amdgcn-amd-amdhsa -S -passes='require<libcall-lowering-info>,atomic-expand' %s | FileCheck %s

; A sub-word (i16) cmpxchg is widened to a word-sized cmpxchg retry loop by
; expandPartwordCmpXchg. Both the seed load (InitLoaded) and the widened cmpxchg
; must preserve the original !tbaa/!noalias metadata, matching the sibling
; widenPartwordAtomicRMW path.

define { i16, i1 } @test_cmpxchg_i16_preserves_metadata(ptr %p, i16 %cmp, i16 %new) {
; CHECK-LABEL: @test_cmpxchg_i16_preserves_metadata(
; CHECK:         load i32, ptr {{%.*}}, align 4{{.*}}!tbaa ![[TBAA:[0-9]+]]{{.*}}!noalias ![[NOALIAS:[0-9]+]]
; CHECK:         cmpxchg ptr {{%.*}} seq_cst seq_cst, align 4{{.*}}!tbaa ![[TBAA]]{{.*}}!noalias ![[NOALIAS]]
  %r = cmpxchg ptr %p, i16 %cmp, i16 %new seq_cst seq_cst, align 2, !tbaa !2, !noalias !6
  ret { i16, i1 } %r
}

!0 = !{!"alias-domain"}
!1 = !{!"alias-scope-a", !0}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!1}
