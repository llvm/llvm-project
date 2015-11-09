; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@m = external global i32, align 4

; Function Attrs: nounwind
define signext i32 @main() #0 {
entry:

; CHECK-LABEL: @main
; CHECK-NOT: rlwimi
; CHECK: andi

  %0 = load i32, i32* @m, align 4
  %or = or i32 %0, 250
  store i32 %or, i32* @m, align 4
  %and = and i32 %or, 249
  %sub.i = sub i32 %and, 0
  %sext = shl i32 %sub.i, 24
  %conv = ashr exact i32 %sext, 24
  ret i32 %conv
}

attributes #0 = { nounwind "target-cpu"="pwr7" }
attributes #1 = { nounwind }

