; RUN: opt -S -passes='dxil-legalize' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define i32 @i8trunc(float %0) #0 {
  ; CHECK-NOT: %4 = trunc nsw i32 %3 to i8
  ; CHECK: add i32
  ; CHECK-NEXT: srem i32
  ; CHECK-NEXT: sub i32
  ; CHECK-NEXT: mul i32
  ; CHECK-NEXT: udiv i32
  ; CHECK-NEXT: sdiv i32
  ; CHECK-NEXT: urem i32
  ; CHECK-NEXT: and i32
  ; CHECK-NEXT: or i32
  ; CHECK-NEXT: xor i32
  ; CHECK-NEXT: shl i32
  ; CHECK-NEXT: lshr i32
  ; CHECK-NEXT: ashr i32
  ; CHECK-NOT: %7 = sext i8 %6 to i32
  
  %2 = fptosi float %0 to i32
  %3 = srem i32 %2, 8
  %4 = trunc nsw i32 %3 to i8
  %5 = add nsw i8 %4, 1
  %6 = srem i8 %5, 8
  %7 = sub i8 %6, 1
  %8 = mul i8 %7, 1
  %9 = udiv i8 %8, 1
  %10 = sdiv i8 %9, 1
  %11 = urem i8 %10, 1
  %12 = and i8 %11, 1
  %13 = or i8 %12, 1
  %14 = xor i8 %13, 1
  %15 = shl i8 %14, 1
  %16 = lshr i8 %15, 1
  %17 = ashr i8 %16, 1
  %18 = sext i8 %17 to i32
  ret i32 %18
}
