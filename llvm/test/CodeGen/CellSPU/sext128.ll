; RUN: llc < %s -march=cellspu | FileCheck %s 

; ModuleID = 'sext128.bc'
target datalayout = "E-p:32:32:128-i1:8:128-i8:8:128-i16:16:128-i32:32:128-i64:32:128-f32:32:128-f64:64:128-v64:128:128-v128:128:128-a0:0:128-s0:128:128"
target triple = "spu"

define i128 @sext_i64_i128(i64 %a) {
entry:
        %0 = sext i64 %a to i128
        ret i128 %0
; CHECK: 	long	269488144
; CHECK: 	long	269488144
; CHECK:	long	66051
; CHECK: 	long	67438087
; CHECK-NOT: rotqmbyi
; CHECK:	lqa
; CHECK: 	rotmai
; CHECK:	shufb
}

define i128 @sext_i32_i128(i32 %a) {
entry:
        %0 = sext i32 %a to i128
        ret i128 %0
; CHECK: 	long	269488144
; CHECK: 	long	269488144
; CHECK: 	long	269488144
; CHECK:	long	66051
; CHECK-NOT: rotqmbyi
; CHECK:	lqa
; CHECK: 	rotmai
; CHECK:	shufb
}

define i128 @sext_i32_i128a(float %a) {
entry:
  %0 = call i32 @myfunc(float %a)
  %1 = sext i32 %0 to i128
  ret i128 %1
; CHECK: 	long	269488144
; CHECK: 	long	269488144
; CHECK: 	long	269488144
; CHECK:	long	66051
; CHECK-NOT: rotqmbyi
; CHECK:	lqa
; CHECK: 	rotmai
; CHECK:	shufb
}

declare i32 @myfunc(float)

define i128 @func1(i8 %u) {
entry:
; CHECK: xsbh
; CHECK: xshw
; CHECK: rotmai
; CHECK: shufb
; CHECK: bi $lr
      %0 = sext i8 %u to i128
      ret i128 %0
}

define i128 @func2(i16 %u) {
entry:
; CHECK: xshw
; CHECK: rotmai
; CHECK: shufb
; CHECK: bi $lr
      %0 = sext i16 %u to i128
      ret i128 %0
}
