; RUN: llc -disable-fp-elim < %s | FileCheck %s
; PR8749
target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f128:64:128-n32"
target triple = "powerpc-apple-darwin9.8"

define i32 @main() nounwind {
entry:
; Make sure we're generating references using the red zone
; CHECK: main:
; CHECK: stw r3, -12(r1)
  %retval = alloca i32
  %0 = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  store i32 0, i32* %0, align 4
  %1 = load i32* %0, align 4
  store i32 %1, i32* %retval, align 4
  br label %return

return:                                           ; preds = %entry
  %retval1 = load i32* %retval
  ret i32 %retval1
}
