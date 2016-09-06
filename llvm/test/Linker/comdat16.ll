; RUN: llvm-link -S -o - %s %p/Inputs/comdat16.ll | FileCheck %s

$c1 = comdat any

@v1 = weak_odr global i32 42, comdat($c1)
define weak_odr i32 @f1(i8*) comdat($c1) {
bb10:
  br label %bb11
bb11:
  ret i32 42
}

@r11 = global i32* @v1
@r12 = global i32 (i8*)* @f1

@a11 = alias i32, i32* @v1
@a12 = alias i16, bitcast (i32* @v1 to i16*)

@a13 = alias i32 (i8*), i32 (i8*)* @f1
@a14 = alias i16, bitcast (i32 (i8*)* @f1 to i16*)
@a15 = alias i16, i16* @a14

; CHECK: $c1 = comdat any
; CHECK: $c2 = comdat any

; CHECK-DAG: @will_be_undefined = external global i32

; CHECK-DAG: @v1 = weak_odr global i32 42, comdat($c1)

; CHECK-DAG: @r11 = global i32* @v1{{$}}
; CHECK-DAG: @r12 = global i32 (i8*)* @f1{{$}}

; CHECK-DAG: @r21 = global i32* @v1{{$}}
; CHECK-DAG: @r22 = global i32 (i8*)* @f1{{$}}

; CHECK-DAG: @v1.1 = internal global i32 41, comdat($c2)

; CHECK-DAG: @a11 = alias i32, i32* @v1{{$}}
; CHECK-DAG: @a12 = alias i16, bitcast (i32* @v1 to i16*)

; CHECK-DAG: @a13 = alias i32 (i8*), i32 (i8*)* @f1{{$}}
; CHECK-DAG: @a14 = alias i16, bitcast (i32 (i8*)* @f1 to i16*)

; CHECK-DAG: @a21 = alias i32, i32* @v1.1{{$}}
; CHECK-DAG: @a22 = alias i16, bitcast (i32* @v1.1 to i16*)

; CHECK-DAG: @a23 = alias i32 (i8*), i32 (i8*)* @f1.2{{$}}
; CHECK-DAG: @a24 = alias i16, bitcast (i32 (i8*)* @f1.2 to i16*)

; CHECK:      define weak_odr protected i32 @f1(i8*) comdat($c1) {
; CHECK-NEXT: bb10:
; CHECK-NEXT:   br label %bb11{{$}}
; CHECK:      bb11:
; CHECK-NEXT:   ret i32 42
; CHECK-NEXT: }

; CHECK:      define internal i32 @f1.2(i8* %this) comdat($c2) {
; CHECK-NEXT: bb20:
; CHECK-NEXT:   store i8* %this, i8** null
; CHECK-NEXT:   br label %bb21
; CHECK:      bb21:
; CHECK-NEXT:   ret i32 41
; CHECK-NEXT: }
