; RUN: opt -S %s -passes='require<libcall-lowering-info>,atomic-expand' -mtriple=x86_64-linux-gnu | FileCheck %s
; RUN: opt -S %s -passes='require<libcall-lowering-info>,atomic-expand' -mtriple=x86_64-linux-gnu -mattr=+cx16 | FileCheck %s --check-prefixes=CHECK,CX16

; AtomicExpand casts float/vector/pointer atomics to integer atomics so the
; backend can select them. The cast must preserve the original !tbaa/!noalias
; metadata, matching the sibling convertAtomicXchgToIntegerType.

; convertAtomicLoadToIntegerType
define float @castload(ptr %p) {
; CHECK-LABEL: define float @castload(
; CHECK:    load atomic i32, ptr {{%.*}} seq_cst, align 4, !tbaa ![[TBAA:[0-9]+]], !noalias ![[NOALIAS:[0-9]+]]
  %r = load atomic float, ptr %p seq_cst, align 4, !tbaa !2, !noalias !6
  ret float %r
}

; convertAtomicStoreToIntegerType
define void @caststore(ptr %p, float %v) {
; CHECK-LABEL: define void @caststore(
; CHECK:    store atomic i32 {{%.*}}, ptr {{%.*}} seq_cst, align 4, !tbaa ![[TBAA]], !noalias ![[NOALIAS]]
  store atomic float %v, ptr %p seq_cst, align 4, !tbaa !2, !noalias !6
  ret void
}

; convertCmpXchgToIntegerType (pointer cmpxchg cast to integer)
define { ptr, i1 } @castcmpxchg(ptr %p, ptr %c, ptr %n) {
; CHECK-LABEL: define { ptr, i1 } @castcmpxchg(
; CHECK:    cmpxchg ptr {{%.*}} seq_cst seq_cst, align 8, !tbaa ![[TBAA]], !noalias ![[NOALIAS]]
  %r = cmpxchg ptr %p, ptr %c, ptr %n seq_cst seq_cst, align 8, !tbaa !2, !noalias !6
  ret { ptr, i1 } %r
}

; expandAtomicLoadToCmpXchg (wide i128 load -> dummy cmpxchg, needs cx16)
define i128 @load_cmpxchg(ptr %p) {
; CX16-LABEL: define i128 @load_cmpxchg(
; CX16:    cmpxchg ptr {{%.*}}, i128 0, i128 0 seq_cst seq_cst, align 16, !tbaa ![[TBAA]], !noalias ![[NOALIAS]]
  %r = load atomic i128, ptr %p seq_cst, align 16, !tbaa !2, !noalias !6
  ret i128 %r
}

!0 = !{!"alias-domain"}
!1 = !{!"alias-scope-a", !0}
!2 = !{!3, !3, i64 0}
!3 = !{!"int", !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}
!6 = !{!1}
