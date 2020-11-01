; RUN: llc -relocation-model=pic < %s -mtriple=ve-unknown-unknown | FileCheck %s

@dst = internal unnamed_addr global i32 0, align 4
@src = internal unnamed_addr global i1 false, align 4
@.str = private unnamed_addr constant [3 x i8] c"%d\00", align 1

define void @func() {
; CHECK-LABEL: func:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; CHECK-NEXT:    and %s15, %s15, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; CHECK-NEXT:    lea %s0, src@gotoff_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, src@gotoff_hi(, %s0)
; CHECK-NEXT:    ld1b.zx %s0, (%s0, %s15)
; CHECK-NEXT:    lea %s1, 100
; CHECK-NEXT:    cmov.w.eq %s1, (0)1, %s0
; CHECK-NEXT:    lea %s0, dst@gotoff_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, dst@gotoff_hi(, %s0)
; CHECK-NEXT:    stl %s1, (%s0, %s15)
; CHECK-NEXT:    or %s11, 0, %s9

  %1 = load i1, i1* @src, align 4
  %2 = select i1 %1, i32 100, i32 0
  store i32 %2, i32* @dst, align 4
  ret void
}

; Function Attrs: nounwind
define i32 @main() {
; CHECK-LABEL: main:
; CHECK:       .LBB{{[0-9]+}}_2:
; CHECK-NEXT:    lea %s15, _GLOBAL_OFFSET_TABLE_@pc_lo(-24)
; CHECK-NEXT:    and %s15, %s15, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s15, _GLOBAL_OFFSET_TABLE_@pc_hi(%s16, %s15)
; CHECK-NEXT:    lea %s0, src@gotoff_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, src@gotoff_hi(, %s0)
; CHECK-NEXT:    or %s1, 1, (0)1
; CHECK-NEXT:    st1b %s1, (%s0, %s15)
; CHECK-NEXT:    lea %s12, func@plt_lo(-24)
; CHECK-NEXT:    and %s12, %s12, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s12, func@plt_hi(%s16, %s12)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    lea %s0, dst@gotoff_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, dst@gotoff_hi(, %s0)
; CHECK-NEXT:    ldl.sx %s1, (%s0, %s15)
; CHECK-NEXT:    st %s1, 184(, %s11)
; CHECK-NEXT:    lea %s0, .L.str@gotoff_lo
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    lea.sl %s0, .L.str@gotoff_hi(%s0, %s15)
; CHECK-NEXT:    lea %s12, printf@plt_lo(-24)
; CHECK-NEXT:    and %s12, %s12, (32)0
; CHECK-NEXT:    sic %s16
; CHECK-NEXT:    lea.sl %s12, printf@plt_hi(%s16, %s12)
; CHECK-NEXT:    st %s0, 176(, %s11)
; CHECK-NEXT:    bsic %s10, (, %s12)
; CHECK-NEXT:    or %s0, 0, (0)1
; CHECK-NEXT:    or %s11, 0, %s9
  store i1 true, i1* @src, align 4
  tail call void @func()
  %1 = load i32, i32* @dst, align 4
  %2 = tail call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @.str, i64 0, i64 0), i32 %1)
  ret i32 0
}

declare i32 @printf(i8* nocapture readonly, ...)
