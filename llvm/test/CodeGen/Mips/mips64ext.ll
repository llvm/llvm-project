; RUN: llc  < %s -mtriple=mips64el -mcpu=mips4 -target-abi=n64 | FileCheck %s
; RUN: llc  < %s -mtriple=mips64el -mcpu=mips64 -target-abi=n64 | FileCheck %s

define i64 @zext64_32(i32 %a) nounwind readnone {
entry:
; CHECK: addiu $[[R0:[0-9]+]], ${{[0-9]+}}, 2
; CHECK: dsll $[[R1:[0-9]+]], $[[R0]], 32
; CHECK: dsrl ${{[0-9]+}}, $[[R1]], 32
  %add = add i32 %a, 2
  %conv = zext i32 %add to i64
  ret i64 %conv
}

define i64 @sext64_32(i32 %a) nounwind readnone {
entry:
; CHECK: sll ${{[0-9]+}}, ${{[0-9]+}}, 0
  %conv = sext i32 %a to i64
  ret i64 %conv
}

define i64 @i64_float(float %f) nounwind readnone {
entry:
; CHECK: trunc.l.s 
  %conv = fptosi float %f to i64
  ret i64 %conv
}

