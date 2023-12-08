; RUN: llc -march=mipsel < %s | FileCheck %s

define i64 @shl0(i64 %a, i32 %b) nounwind readnone {
entry:
; CHECK: shl0
; CHECK-NOT: lw $25, %call16(__
  %sh_prom = zext i32 %b to i64
  %shl = shl i64 %a, %sh_prom
  ret i64 %shl
}

define i64 @shr1(i64 %a, i32 %b) nounwind readnone {
entry:
; CHECK: shr1
; CHECK-NOT: lw $25, %call16(__
  %sh_prom = zext i32 %b to i64
  %shr = lshr i64 %a, %sh_prom
  ret i64 %shr
}

define i64 @sra2(i64 %a, i32 %b) nounwind readnone {
entry:
; CHECK: sra2
; CHECK-NOT: lw $25, %call16(__
  %sh_prom = zext i32 %b to i64
  %shr = ashr i64 %a, %sh_prom
  ret i64 %shr
}

