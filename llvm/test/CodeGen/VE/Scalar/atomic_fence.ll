; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test atomic fence for all memory order

; Function Attrs: norecurse nounwind readnone
define void @_Z20atomic_fence_relaxedv() {
; CHECK-LABEL: _Z20atomic_fence_relaxedv:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    or %s11, 0, %s9
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_consumev() {
; CHECK-LABEL: _Z20atomic_fence_consumev:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    or %s11, 0, %s9
  fence acquire
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_acquirev() {
; CHECK-LABEL: _Z20atomic_fence_acquirev:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    or %s11, 0, %s9
  fence acquire
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_releasev() {
; CHECK-LABEL: _Z20atomic_fence_releasev:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    or %s11, 0, %s9
  fence release
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_acq_relv() {
; CHECK-LABEL: _Z20atomic_fence_acq_relv:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    or %s11, 0, %s9
  fence acq_rel
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_seq_cstv() {
; CHECK-LABEL: _Z20atomic_fence_seq_cstv:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    or %s11, 0, %s9
  fence seq_cst
  ret void
}
