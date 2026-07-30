; RUN: llc -relocation-model=pic < %s -mtriple=ve-unknown-unknown | FileCheck %s

@dst = external global i32, align 4
@ptr = external global ptr, align 8
@src = external global i32, align 4

define i32 @func() {
; CHECK-LABEL: func:
; CHECK:       # %bb.0:
; CHECK-NEXT:    st %s15, 24(, %s11)
; CHECK-NEXT:    st %s16, 32(, %s11)
; CHECK-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; CHECK-NEXT:    and %s15, %s15, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; CHECK-NEXT:    lea %s0, dst@got_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, dst@got_hi(, %s0)
; CHECK-NEXT:    ld %s1, (%s0, %s15)
; CHECK-NEXT:    lea %s0, ptr@got_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea %s2, src@got_lo
; CHECK-NEXT:    and %s2, %s2, (32)0
; CHECK-NEXT:    lea.sl %s2, src@got_hi(, %s2)
; CHECK-NEXT:    ld %s2, (%s2, %s15)
; CHECK-NEXT:    lea.sl %s0, ptr@got_hi(, %s0)
; CHECK-NEXT:    ld %s0, (%s0, %s15)
; CHECK-NEXT:    ldl.sx %s2, (, %s2)
; CHECK-NEXT:    st %s1, (, %s0)
; CHECK-NEXT:    or %s0, 1, (0)1
; CHECK-NEXT:    stl %s2, (, %s1)
; CHECK-NEXT:    ld %s16, 32(, %s11)
; CHECK-NEXT:    ld %s15, 24(, %s11)
; CHECK-NEXT:    b.l.t (, %s10)

  store ptr @dst, ptr @ptr, align 8
  %1 = load i32, ptr @src, align 4
  store i32 %1, ptr @dst, align 4
  ret i32 1
}
