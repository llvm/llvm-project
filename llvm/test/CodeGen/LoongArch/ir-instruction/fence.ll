; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s --check-prefix=LA64

define void @fence_acquire() nounwind {
; LA32-LABEL: fence_acquire:
; LA32:       # %bb.0:
; LA32-NEXT:    dbar 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fence_acquire:
; LA64:       # %bb.0:
; LA64-NEXT:    dbar 0
; LA64-NEXT:    jirl $zero, $ra, 0
  fence acquire
  ret void
}

define void @fence_release() nounwind {
; LA32-LABEL: fence_release:
; LA32:       # %bb.0:
; LA32-NEXT:    dbar 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fence_release:
; LA64:       # %bb.0:
; LA64-NEXT:    dbar 0
; LA64-NEXT:    jirl $zero, $ra, 0
  fence release
  ret void
}

define void @fence_acq_rel() nounwind {
; LA32-LABEL: fence_acq_rel:
; LA32:       # %bb.0:
; LA32-NEXT:    dbar 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fence_acq_rel:
; LA64:       # %bb.0:
; LA64-NEXT:    dbar 0
; LA64-NEXT:    jirl $zero, $ra, 0
  fence acq_rel
  ret void
}

define void @fence_seq_cst() nounwind {
; LA32-LABEL: fence_seq_cst:
; LA32:       # %bb.0:
; LA32-NEXT:    dbar 0
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: fence_seq_cst:
; LA64:       # %bb.0:
; LA64-NEXT:    dbar 0
; LA64-NEXT:    jirl $zero, $ra, 0
  fence seq_cst
  ret void
}
