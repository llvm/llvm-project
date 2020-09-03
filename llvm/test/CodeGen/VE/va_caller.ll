; RUN: llc < %s -mtriple=ve-unknown-unknown | FileCheck %s

declare i32 @func(i32, ...)

define i32 @caller() {
; CHECK-LABEL: caller:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    st %s18, 48(, %s9) # 8-byte Folded Spill
; CHECK-NEXT:    or %s7, 0, (0)1
; CHECK-NEXT:    st %s7, 280(, %s11)
; CHECK-NEXT:    or %s0, 11, (0)1
; CHECK-NEXT:    st %s0, 272(, %s11)
; CHECK-NEXT:    st %s7, 264(, %s11)
; CHECK-NEXT:    or %s0, 10, (0)1
; CHECK-NEXT:    st %s0, 256(, %s11)
; CHECK-NEXT:    lea.sl %s0, 1075970048
; CHECK-NEXT:    st %s0, 248(, %s11)
; CHECK-NEXT:    or %s0, 8, (0)1
; CHECK-NEXT:    st %s0, 240(, %s11)
; CHECK-NEXT:    st %s7, 232(, %s11)
; CHECK-NEXT:    lea %s0, 1086324736
; CHECK-NEXT:    stl %s0, 228(, %s11)
; CHECK-NEXT:    or %s5, 5, (0)1
; CHECK-NEXT:    stl %s5, 216(, %s11)
; CHECK-NEXT:    or %s4, 4, (0)1
; CHECK-NEXT:    stl %s4, 208(, %s11)
; CHECK-NEXT:    or %s3, 3, (0)1
; CHECK-NEXT:    stl %s3, 200(, %s11)
; CHECK-NEXT:    or %s2, 2, (0)1
; CHECK-NEXT:    stl %s2, 192(, %s11)
; CHECK-NEXT:    or %s1, 1, (0)1
; CHECK-NEXT:    stl %s1, 184(, %s11)
; CHECK-NEXT:    or %s18, 0, (0)1
; CHECK-NEXT:    lea %s0, func@lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s12, func@hi(, %s0)
; CHECK-NEXT:    lea.sl %s6, 1086324736
; CHECK-NEXT:    stl %s18, 176(, %s11)
; CHECK-NEXT:    or %s0, 0, %s18
; CHECK-NEXT:    # kill: def $sf6 killed $sf6 killed $sx6
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s0, 0, %s18
; CHECK-NEXT:    ld %s18, 48(, %s9) # 8-byte Folded Reload
; CHECK-NEXT:    or %s11, 0, %s9
  call i32 (i32, ...) @func(i32 0, i16 1, i8 2, i32 3, i16 4, i8 5, float 6.0, i8* null, i64 8, double 9.0, i128 10, i128 11)
  ret i32 0
}
