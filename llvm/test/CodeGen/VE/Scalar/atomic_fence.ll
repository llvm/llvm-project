; RUN: llc < %s -mtriple=ve | FileCheck %s

;;; Test atomic fence for all memory order

; Function Attrs: norecurse nounwind readnone
define void @_Z20atomic_fence_relaxedv() {
; CHECK-LABEL: _Z20atomic_fence_relaxedv:
; CHECK:       # %bb.0:
; CHECK-NEXT:    b.l.t (, %s10)
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_consumev() {
; CHECK-LABEL: _Z20atomic_fence_consumev:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  fence acquire
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_acquirev() {
; CHECK-LABEL: _Z20atomic_fence_acquirev:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 2
; CHECK-NEXT:    b.l.t (, %s10)
  fence acquire
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_releasev() {
; CHECK-LABEL: _Z20atomic_fence_releasev:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 1
; CHECK-NEXT:    b.l.t (, %s10)
  fence release
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_acq_relv() {
; CHECK-LABEL: _Z20atomic_fence_acq_relv:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  fence acq_rel
  ret void
}

; Function Attrs: nofree norecurse nounwind
define void @_Z20atomic_fence_seq_cstv() {
; CHECK-LABEL: _Z20atomic_fence_seq_cstv:
; CHECK:       # %bb.0:
; CHECK-NEXT:    fencem 3
; CHECK-NEXT:    b.l.t (, %s10)
  fence seq_cst
  ret void
}
