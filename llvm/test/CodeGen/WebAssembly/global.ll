; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that globals assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

; CHECK-NOT: llvm.used
; CHECK-NOT: llvm.metadata
@llvm.used = appending global [1 x i32*] [i32* @g], section "llvm.metadata"

; CHECK: foo:
; CHECK: i32.const $push0=, answer{{$}}
; CHECK-NEXT: i32.load $0=, $pop0{{$}}
; CHECK-NEXT: return $0{{$}}
define i32 @foo() {
  %a = load i32, i32* @answer
  ret i32 %a
}

; CHECK: .type   g,@object
; CHECK: .align  2{{$}}
; CHECK-NEXT: g:
; CHECK-NEXT: .int32 1337{{$}}
; CHECK-NEXT: .size g, 4{{$}}
@g = private global i32 1337

; CHECK-LABEL: ud:
; CHECK-NEXT: .zero 4{{$}}
; CHECK-NEXT: .size ud, 4{{$}}
@ud = internal global i32 undef

; CHECK: .type nil,@object
; CHECK-NEXT: .lcomm nil,4,2{{$}}
@nil = internal global i32 zeroinitializer

; CHECK: .type z,@object
; CHECK-NEXT: .lcomm z,4,2{{$}}
@z = internal global i32 0

; CHECK-NEXT: .type one,@object
; CHECK-NEXT: .align 2{{$}}
; CHECK-NEXT: one:
; CHECK-NEXT: .int32 1{{$}}
; CHECK-NEXT: .size one, 4{{$}}
@one = internal global i32 1

; CHECK: .type answer,@object
; CHECK: .align 2{{$}}
; CHECK-NEXT: answer:
; CHECK-NEXT: .int32 42{{$}}
; CHECK-NEXT: .size answer, 4{{$}}
@answer = internal global i32 42

; CHECK: .type u32max,@object
; CHECK: .align 2{{$}}
; CHECK-NEXT: u32max:
; CHECK-NEXT: .int32 4294967295{{$}}
; CHECK-NEXT: .size u32max, 4{{$}}
@u32max = internal global i32 -1

; CHECK: .type ud64,@object
; CHECK: .align 3{{$}}
; CHECK-NEXT: ud64:
; CHECK-NEXT: .zero 8{{$}}
; CHECK-NEXT: .size ud64, 8{{$}}
@ud64 = internal global i64 undef

; CHECK: .type nil64,@object
; CHECK: .lcomm nil64,8,3{{$}}
@nil64 = internal global i64 zeroinitializer

; CHECK: .type z64,@object
; CHECK: .lcomm z64,8,3{{$}}
@z64 = internal global i64 0

; CHECK: .type twoP32,@object
; CHECK: .align 3{{$}}
; CHECK-NEXT: twoP32:
; CHECK-NEXT: .int64 4294967296{{$}}
; CHECK-NEXT: .size twoP32, 8{{$}}
@twoP32 = internal global i64 4294967296

; CHECK: .type u64max,@object
; CHECK: .align 3{{$}}
; CHECK-NEXT: u64max:
; CHECK-NEXT: .int64 -1{{$}}
; CHECK-NEXT: .size u64max, 8{{$}}
@u64max = internal global i64 -1

; CHECK: .type f32ud,@object
; CHECK: .align 2{{$}}
; CHECK-NEXT: f32ud:
; CHECK-NEXT: .zero 4{{$}}
; CHECK-NEXT: .size f32ud, 4{{$}}
@f32ud = internal global float undef

; CHECK: .type f32nil,@object
; CHECK: .lcomm f32nil,4,2{{$}}
@f32nil = internal global float zeroinitializer

; CHECK: .type f32z,@object
; CHECK: .lcomm f32z,4,2{{$}}
@f32z = internal global float 0.0

; CHECK: .type f32nz,@object
; CHECK: .align 2{{$}}
; CHECK: f32nz:
; CHECK: .int32 2147483648{{$}}
; CHECK: .size f32nz, 4{{$}}
@f32nz = internal global float -0.0

; CHECK: .type f32two,@object
; CHECK: .align 2{{$}}
; CHECK-NEXT: f32two:
; CHECK-NEXT: .int32 1073741824{{$}}
; CHECK-NEXT: .size f32two, 4{{$}}
@f32two = internal global float 2.0

; CHECK: .type f64ud,@object
; CHECK: .align 3{{$}}
; CHECK-NEXT: f64ud:
; CHECK-NEXT: .zero 8{{$}}
; CHECK-NEXT: .size f64ud, 8{{$}}
@f64ud = internal global double undef

; CHECK: .type f64nil,@object
; CHECK: .lcomm f64nil,8,3{{$}}
@f64nil = internal global double zeroinitializer

; CHECK: .type f64z,@object
; CHECK: .lcomm f64z,8,3{{$}}
@f64z = internal global double 0.0

; CHECK: .type f64nz,@object
; CHECK: .align 3{{$}}
; CHECK-NEXT: f64nz:
; CHECK-NEXT: .int64 -9223372036854775808{{$}}
; CHECK-NEXT: .size f64nz, 8{{$}}
@f64nz = internal global double -0.0

; CHECK: .type f64two,@object
; CHECK: .align 3{{$}}
; CHECK-NEXT: f64two:
; CHECK-NEXT: .int64 4611686018427387904{{$}}
; CHECK-NEXT: .size f64two, 8{{$}}
@f64two = internal global double 2.0
