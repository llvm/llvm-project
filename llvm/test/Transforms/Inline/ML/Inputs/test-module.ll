target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

declare void @external_fct(i32)

define dso_local i32 @top() {
  %a = call i32 @multiplier(i32 5)
  %b = call i32 @adder(i32 10)
  %ret = add nsw i32 %a, %b
  call void @external_fct(i32 %ret)
  ret i32 %ret
}

define internal dso_local i32 @adder(i32) {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = call i32 @multiplier(i32 %3)
  %5 = load i32, i32* %2, align 4
  %6 = call i32 @switcher(i32 1)
  %7 = add nsw i32 %4, %6
  ret i32 %7
}

define internal i32 @multiplier(i32) {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = load i32, i32* %2, align 4
  %5 = mul nsw i32 %3, %4
  ret i32 %5
}

define i32 @switcher(i32) {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  %4 = load i32, i32* %3, align 4
  switch i32 %4, label %11 [
    i32 1, label %5
    i32 2, label %6
  ]

; <label>:5:                                      ; preds = %1
  store i32 2, i32* %2, align 4
  br label %12

; <label>:6:                                      ; preds = %1
  %7 = load i32, i32* %3, align 4
  %8 = load i32, i32* %3, align 4
  %9 = call i32 @multiplier(i32 %8)
  %10 = add nsw i32 %7, %9
  store i32 %10, i32* %2, align 4
  br label %12

; <label>:11:                                     ; preds = %1
  %adder.result = call i32 @adder(i32 2)
  store i32 %adder.result, i32* %2, align 4
  br label %12

; <label>:12:                                     ; preds = %11, %6, %5
  %13 = load i32, i32* %2, align 4
  ret i32 %13
}

; CHECK-NOT: @adder
; DEFAULT-LABEL:        @adder
; DEFAULT-NEXT:         %2 = mul