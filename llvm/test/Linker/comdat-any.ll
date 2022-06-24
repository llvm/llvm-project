; RUN: rm -rf %t && split-file %s %t
; RUN: llvm-link -S %t/1.ll %t/1-aux.ll -o - | FileCheck %s
; RUN: llvm-link -S %t/2.ll %t/2-aux.ll -o - | FileCheck %s --check-prefix=CHECK2

;; A private linkage variable can be copied.
; RUN: llvm-link -S %t/3.ll %t/3-aux.ll -o - | FileCheck %s --check-prefix=CHECK3

;--- 1.ll
; CHECK: @foo = global i8 0, comdat

; CHECK: define void @zed() {
; CHECK:   call void @bar()
; CHECK:   ret void
; CHECK: }

; CHECK: declare void @bar()

$foo = comdat any
@foo = global i8 0, comdat

;--- 1-aux.ll
$foo = comdat any
@foo = global i8 1, comdat
define void @zed() {
  call void @bar()
  ret void
}
define void @bar() comdat($foo) {
  ret void
}

;--- 2.ll
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

; CHECK2: $c1 = comdat any
; CHECK2: $c2 = comdat any

; CHECK2-DAG: @will_be_undefined = external global i32

; CHECK2-DAG: @v1 = weak_odr global i32 42, comdat($c1)

; CHECK2-DAG: @r11 = global i32* @v1{{$}}
; CHECK2-DAG: @r12 = global i32 (i8*)* @f1{{$}}

; CHECK2-DAG: @r21 = global i32* @v1{{$}}
; CHECK2-DAG: @r22 = global i32 (i8*)* @f1{{$}}

; CHECK2-DAG: @v1.1 = internal global i32 41, comdat($c2)

; CHECK2-DAG: @a11 = alias i32, i32* @v1{{$}}
; CHECK2-DAG: @a12 = alias i16, bitcast (i32* @v1 to i16*)

; CHECK2-DAG: @a13 = alias i32 (i8*), i32 (i8*)* @f1{{$}}
; CHECK2-DAG: @a14 = alias i16, bitcast (i32 (i8*)* @f1 to i16*)

; CHECK2-DAG: @a21 = alias i32, i32* @v1.1{{$}}
; CHECK2-DAG: @a22 = alias i16, bitcast (i32* @v1.1 to i16*)

; CHECK2-DAG: @a23 = alias i32 (i8*), i32 (i8*)* @f1.2{{$}}
; CHECK2-DAG: @a24 = alias i16, bitcast (i32 (i8*)* @f1.2 to i16*)

; CHECK2:      define weak_odr protected i32 @f1(i8* %0) comdat($c1) {
; CHECK2-NEXT: bb10:
; CHECK2-NEXT:   br label %bb11{{$}}
; CHECK2:      bb11:
; CHECK2-NEXT:   ret i32 42
; CHECK2-NEXT: }

; CHECK2:      define internal i32 @f1.2(i8* %this) comdat($c2) {
; CHECK2-NEXT: bb20:
; CHECK2-NEXT:   store i8* %this, i8** null
; CHECK2-NEXT:   br label %bb21
; CHECK2:      bb21:
; CHECK2-NEXT:   ret i32 41
; CHECK2-NEXT: }

;--- 2-aux.ll
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

;--- 3.ll
; CHECK3: @bar = global i32 0, comdat($a1)
; CHECK3: @baz = private global i32 42, comdat($a1)
; CHECK3: @a1 = internal alias i32, i32* @baz
$a1 = comdat any
@bar = global i32 0, comdat($a1)

;--- 3-aux.ll
$a1 = comdat any
@baz = private global i32 42, comdat($a1)
@a1 = internal alias i32, i32* @baz
define i32* @abc() {
  ret i32* @a1
}
