; RUN: llvm-as %s -o %t.o
; RUN: llvm-as %p/Inputs/comdat.ll -o %t2.o
; RUN: llvm-lto2 -save-temps -o %t3.o %t.o %t2.o \
; RUN:  -r=%t.o,f1,plx \
; RUN:  -r=%t.o,v1,px \
; RUN:  -r=%t.o,r11,px \
; RUN:  -r=%t.o,r12,px \
; RUN:  -r=%t.o,a11,px \
; RUN:  -r=%t.o,a12,px \
; RUN:  -r=%t.o,a13,px \
; RUN:  -r=%t.o,a14,px \
; RUN:  -r=%t.o,a15,px \
; RUN:  -r=%t2.o,f1,l \
; RUN:  -r=%t2.o,will_be_undefined, \
; RUN:  -r=%t2.o,v1, \
; RUN:  -r=%t2.o,r21,px \
; RUN:  -r=%t2.o,r22,px \
; RUN:  -r=%t2.o,a21,px \
; RUN:  -r=%t2.o,a22,px \
; RUN:  -r=%t2.o,a23,px \
; RUN:  -r=%t2.o,a24,px \
; RUN:  -r=%t2.o,a25,px
; RUN: llvm-dis %t3.o.0.2.internalize.bc -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

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

; CHECK:      define weak_odr i32 @f1(i8*) comdat($c1) {
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
