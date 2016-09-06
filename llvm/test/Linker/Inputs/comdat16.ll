$c2 = comdat any
$c1 = comdat any

; This is only present in this file. The linker will keep $c1 from the first
; file and this will be undefined.
@will_be_undefined = global i32 1, comdat($c1)
@use = global i32* @will_be_undefined

@v1 = weak_odr global i32 41, comdat($c2)
define weak_odr protected i32 @f1(i8* %this) comdat($c2) {
bb20:
  store i8* %this, i8** null
  br label %bb21
bb21:
  ret i32 41
}

@r21 = global i32* @v1
@r22 = global i32(i8*)* @f1

@a21 = alias i32, i32* @v1
@a22 = alias i16, bitcast (i32* @v1 to i16*)

@a23 = alias i32(i8*), i32(i8*)* @f1
@a24 = alias i16, bitcast (i32(i8*)* @f1 to i16*)
@a25 = alias i16, i16* @a24
