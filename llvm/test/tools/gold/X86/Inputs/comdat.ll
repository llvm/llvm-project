target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

$c2 = comdat any
$c1 = comdat any

; This is only present in this file. The linker will keep $c1 from the first
; file and this will be undefined.
@will_be_undefined = global i32 1, comdat($c1)

@v1 = weak_odr global i32 41, comdat($c2)
define weak_odr protected i32 @f1(ptr %this) comdat($c2) {
bb20:
  store ptr %this, ptr null
  br label %bb21
bb21:
  ret i32 41
}

@r21 = global ptr @v1
@r22 = global ptr @f1

@a21 = alias i32, ptr @v1
@a22 = alias i16, ptr @v1

@a23 = alias i32(ptr), ptr @f1
@a24 = alias i16, ptr @f1
@a25 = alias i16, ptr @a24
