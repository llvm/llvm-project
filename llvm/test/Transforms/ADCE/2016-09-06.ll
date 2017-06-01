; RUN: opt < %s -sroa -adce -adce-remove-loops -S | FileCheck %s
; ModuleID = 'test1.bc'
source_filename = "test1.c"
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind uwtable
define i32 @foo(i32, i32, i32) #0 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  store i32 %0, i32* %4, align 4
  store i32 %1, i32* %5, align 4
  store i32 %2, i32* %6, align 4
  store i32 0, i32* %7, align 4
  %9 = load i32, i32* %5, align 4
  %I10 = icmp ne i32 %9, 0
  br i1 %I10, label %B11, label %B21

B11: 
  store i32 0, i32* %8, align 4
  br label %B12

B12:
  %I13 = load i32, i32* %8, align 4
  %I14 = load i32, i32* %6, align 4
  %I15 = icmp slt i32 %I13, %I14
; CHECK: br label %B20
  br i1 %I15, label %B16, label %B20

B16: 
  br label %B17

B17: 
  %I18 = load i32, i32* %8, align 4
  %I19 = add nsw i32 %I18, 1
  store i32 %I19, i32* %8, align 4
  br label %B12

B20:
  store i32 1, i32* %7, align 4
  br label %B21

B21: 
  %I22 = load i32, i32* %7, align 4
  ret i32 %I22
}

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 4.0.0 (http://llvm.org/git/clang.git 5864a13abf4490e76ae2eb0896198e1305927df2)"}
