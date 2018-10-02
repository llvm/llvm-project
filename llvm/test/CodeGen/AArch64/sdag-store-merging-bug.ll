; RUN: llc -o - %s -mtriple aarch64-- -mattr +slow-misaligned-128store -stop-after=instruction-select | FileCheck %s
; Checks for a bug where selection dag store merging would construct wrong
; indices when extracting values from vectors, resulting in an invalid
; lane duplication in this case.
; The only way I could trigger stores with mismatching types getting merged was
; via the aarch64 slow-misaligned-128store code splitting stores earlier.

; CHECK-LABEL: name: func
; CHECK: LDRQui
; CHECK-NOT: INSERT_SUBREG
; CHECK-NOT: DUP
; CHECK-NEXT: STRQui
define void @func(<2 x double>* %sptr, <2 x double>* %dptr) {
  %load = load <2 x double>, <2 x double>* %sptr, align 8
  ; aarch64 feature slow-misaligned-128store splits the following store.
  ; store merging immediately merges it back together (but used to get the
  ; merging wrong), this is the only way I was able to reproduce the bug...
  store <2 x double> %load, <2 x double>* %dptr, align 4
  ret void
}
