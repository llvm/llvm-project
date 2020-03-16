; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 -disable-mips-delay-filler %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s

; RUN: llc -march=mips64el -filetype=obj -mcpu=mips64r2 %s -o - \
; RUN:   | llvm-objdump -d - | FileCheck %s

define i64 @f3(i64 %a0) nounwind readnone {
entry:
; CHECK: dsll ${{[0-9]+}}, ${{[0-9]+}}, 10
  %shl = shl i64 %a0, 10
  ret i64 %shl
}

define i64 @f4(i64 %a0) nounwind readnone {
entry:
; CHECK: dsra ${{[0-9]+}}, ${{[0-9]+}}, 10
  %shr = ashr i64 %a0, 10
  ret i64 %shr
}

define i64 @f5(i64 %a0) nounwind readnone {
entry:
; CHECK: dsrl ${{[0-9]+}}, ${{[0-9]+}}, 10
  %shr = lshr i64 %a0, 10
  ret i64 %shr
}

define i64 @f6(i64 %a0) nounwind readnone {
entry:
; CHECK: dsll32 ${{[0-9]+}}, ${{[0-9]+}}, 8
  %shl = shl i64 %a0, 40
  ret i64 %shl
}

define i64 @f7(i64 %a0) nounwind readnone {
entry:
; CHECK: dsra32 ${{[0-9]+}}, ${{[0-9]+}}, 8
  %shr = ashr i64 %a0, 40
  ret i64 %shr
}

define i64 @f8(i64 %a0) nounwind readnone {
entry:
; CHECK: dsrl32 ${{[0-9]+}}, ${{[0-9]+}}, 8
  %shr = lshr i64 %a0, 40
  ret i64 %shr
}

