; RUN: opt -S -passes='dxil-legalize-i8' -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

define i32 @i8trunc(float %0) #0 {
  ; CHECK-NOT: %4 = trunc nsw i32 %3 to i8
  ; CHECK: add i32
  ; CHECK: srem i32
  ; CHECK-NOT: %7 = sext i8 %6 to i32
  
  %2 = fptosi float %0 to i32
  %3 = srem i32 %2, 8
  %4 = trunc nsw i32 %3 to i8
  %5 = add nsw i8 %4, 1
  %6 = srem i8 %5, 8
  %7 = sext i8 %6 to i32
  ret i32 %7
}