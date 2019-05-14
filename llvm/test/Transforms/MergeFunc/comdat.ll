; RUN: opt -S -mergefunc %s | FileCheck %s

@symbols = linkonce_odr global <{ i8*, i8* }> <{ i8* bitcast (i32 (i32, i32)* @f to i8*), i8* bitcast (i32 (i32, i32)* @g to i8*) }>

$f = comdat any
$g = comdat any

define linkonce_odr hidden i32 @f(i32 %x, i32 %y) comdat {
  %sum = add i32 %x, %y
  %sum2 = add i32 %x, %sum
  %sum3 = add i32 %x, %sum
  ret i32 %sum3
}

define linkonce_odr hidden i32 @g(i32 %x, i32 %y) comdat {
  %sum = add i32 %x, %y
  %sum2 = add i32 %x, %sum
  %sum3 = add i32 %x, %sum
  ret i32 %sum3
}

; CHECK-DAG: define linkonce_odr hidden i32 @f(i32 %x, i32 %y) comdat
; CHECK-DAG: define linkonce_odr hidden i32 @g(i32, i32) comdat

