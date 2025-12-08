; RUN: llc -mtriple=mipsel -mcpu=mips32r6 -mips-compact-branches=never < %s | FileCheck %s
; RUN: llc -mtriple=mips64el -mcpu=mips64r6 -mips-compact-branches=never < %s | FileCheck %s

;; Test checking we respect mips-compact-branches=never
;; The patterns set + branch should be disabled and not emit compact branches

; CHECK-NOT: bltc
; CHECK-NOT: bgec
; CHECK-NOT: bltuc
; CHECK-NOT: bgeuc
; CHECK-NOT: beqzc
; CHECK-NOT: bnezc


define void @test_slt_never(i32 %a, i32 %b) {
; CHECK-LABEL: test_slt_never:
; CHECK: slt
; CHECK: beqz ${{[0-9]+}}
  %c = icmp slt i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

define void @test_ult_never(i32 %a, i32 %b) {
; CHECK-LABEL: test_ult_never:
; CHECK: sltu
; CHECK: beqz ${{[0-9]+}}
  %c = icmp ult i32 %a, %b
  br i1 %c, label %t, label %f
t:
  ret void
f:
  ret void
}

