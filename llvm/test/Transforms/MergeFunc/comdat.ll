; RUN: opt -S -passes=mergefunc %s | FileCheck %s

@symbols = linkonce_odr global <{ ptr, ptr }> <{ ptr @f, ptr @g }>

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

; CHECK-DAG: define private i32 @0(i32 %x, i32 %y) comdat($f)
; CHECK-DAG: define linkonce_odr hidden i32 @g(i32 %0, i32 %1) comdat {
; CHECK-DAG: define linkonce_odr hidden i32 @f(i32 %0, i32 %1) {

