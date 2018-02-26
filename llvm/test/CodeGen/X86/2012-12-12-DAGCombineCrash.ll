; RUN: llc -mtriple=i686-apple-ios -mcpu=yonah < %s
; rdar://12868039

define void @t() nounwind ssp {
  %1 = alloca i32
  %2 = ptrtoint i32* %1 to i32
  br label %3

; <label>:3                                       ; preds = %5, %3, %0
  switch i32 undef, label %3 [
    i32 611946160, label %5
    i32 954117870, label %4
  ]

; <label>:4                                       ; preds = %3
  ret void

; <label>:5                                       ; preds = %5, %3
  %6 = add i32 0, 148
  %7 = and i32 %6, 48
  %8 = add i32 %7, 0
  %9 = or i32 %2, %8
  %10 = xor i32 -1, %2
  %11 = or i32 %8, %10
  %12 = or i32 %9, %11
  %13 = xor i32 %9, %11
  %14 = sub i32 %12, %13
  %15 = xor i32 2044674005, %14
  %16 = xor i32 %15, 0
  %17 = shl nuw nsw i32 %16, 1
  %18 = sub i32 0, %17
  %19 = and i32 %18, 2051242402
  %20 = sub i32 0, %19
  %21 = xor i32 %20, 0
  %22 = xor i32 %21, 0
  %23 = add i32 0, %22
  %24 = shl i32 %23, 1
  %25 = or i32 1, %24
  %26 = add i32 0, %25
  %27 = trunc i32 %26 to i8
  %28 = xor i8 %27, 125
  %29 = add i8 %28, -16
  %30 = add i8 0, %29
  store i8 %30, i8* null
  br i1 undef, label %5, label %3
}
