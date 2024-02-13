; RUN: llc -relocation-model=pic < %s -mtriple=ve-unknown-unknown | FileCheck %s

@ptr = external global ptr, align 8

define void @func() {
; CHECK-LABEL: func:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; CHECK-NEXT:    and %s15, %s15, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; CHECK-NEXT:    lea %s0, function@got_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, function@got_hi(, %s0)
; CHECK-NEXT:    ld %s0, (%s0, %s15)
; CHECK-NEXT:    lea %s1, ptr@got_lo
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    lea.sl %s1, ptr@got_hi(, %s1)
; CHECK-NEXT:    ld %s1, (%s1, %s15)
; CHECK-NEXT:    st %s0, (, %s1)
; CHECK-NEXT:    or %s12, 0, %s0
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s11, 0, %s9

  store ptr @function, ptr @ptr, align 8
  %1 = load ptr, ptr @ptr, align 8
  %2 = bitcast ptr %1 to ptr
  call void %2()
  ret void
}

declare void @function(...)
