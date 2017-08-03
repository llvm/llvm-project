; RUN: llc < %s -march=x86-64 | FileCheck %s
; PR11495

; CHECK: 1311768467463790320
@v = global <2 x float> bitcast (<1 x i64> <i64 1311768467463790320> to <2 x float>), align 8
