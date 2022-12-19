; RUN: llc -relocation-model=pic < %s -mtriple=ve-unknown-unknown | FileCheck %s

define void @func() {
; CHECK-LABEL: func:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; CHECK-NEXT:    and %s15, %s15, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; CHECK-NEXT:    lea %s12, function@plt_lo(-24)
; CHECK-NEXT:    and %s12, %s12, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s12, function@plt_hi(%s16, %s12)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9

  call void @function()
  ret void
}

declare void @function(...)
