; RUN: llc --mtriple=loongarch64 -mattr=+d --stop-before loongarch-prera-expand-pseudo \
; RUN:     --verify-machineinstrs < %s | FileCheck %s --check-prefix=NOEXPAND
; RUN: llc --mtriple=loongarch64 --stop-after loongarch-prera-expand-pseudo \
; RUN:     --verify-machineinstrs < %s | FileCheck %s --check-prefix=EXPAND

declare void @callee()

define void @caller() nounwind {
; NOEXPAND-LABEL: name: caller
; NOEXPAND: PseudoCALL target-flags{{.*}}callee
;
; EXPAND-LABEL: name: caller
; EXPAND: BL target-flags{{.*}}callee
  call void @callee()
  ret void
}
