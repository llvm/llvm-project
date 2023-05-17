; RUN: opt < %s -wasm-lower-em-ehsjlj -enable-emscripten-sjlj -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

; Tests if an alias to a function (here malloc) is correctly handled as a
; function that cannot longjmp.

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }
@malloc = weak alias ptr (i32), ptr @dlmalloc

; CHECK-LABEL: @malloc_test
define void @malloc_test() {
entry:
  ; CHECK: call ptr @malloc
  %retval = alloca i32, align 4
  %jmp = alloca [1 x %struct.__jmp_buf_tag], align 16
  store i32 0, ptr %retval, align 4
  %call = call i32 @setjmp(ptr %jmp) #0
  call void @foo()
  ret void
}

; This is a dummy dlmalloc implemenation only to make compiler pass, because an
; alias (malloc) has to point an actual definition.
define ptr @dlmalloc(i32) {
  %p = inttoptr i32 0 to ptr
  ret ptr %p
}

declare void @foo()
; Function Attrs: returns_twice
declare i32 @setjmp(ptr) #0

attributes #0 = { returns_twice }
