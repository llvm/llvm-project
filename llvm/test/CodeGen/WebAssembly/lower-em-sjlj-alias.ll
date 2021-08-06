; RUN: opt < %s -wasm-lower-em-ehsjlj -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-emscripten"

; Tests if an alias to a function (here malloc) is correctly handled as a
; function that cannot longjmp.

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }
@malloc = weak alias i8* (i32), i8* (i32)* @dlmalloc

; CHECK-LABEL: @malloc_test
define void @malloc_test() {
entry:
; CHECK-LABEL: entry
  ; All setjmp table preparations have to happen within the entry block. These
  ; check lines list only some of the instructions for that.
  ; CHECK: call i8* @malloc
  ; CHECK: call i32* @saveSetjmp
  ; CHECK: call i32 @getTempRet0
  %retval = alloca i32, align 4
  %jmp = alloca [1 x %struct.__jmp_buf_tag], align 16
  store i32 0, i32* %retval, align 4
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %jmp, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  call void @foo()
  ret void

; CHECK-LABEL: entry.split
}

; This is a dummy dlmalloc implemenation only to make compiler pass, because an
; alias (malloc) has to point an actual definition.
define i8* @dlmalloc(i32) {
  %p = inttoptr i32 0 to i8*
  ret i8* %p
}

declare void @foo()
; Function Attrs: returns_twice
declare i32 @setjmp(%struct.__jmp_buf_tag*) #0

attributes #0 = { returns_twice }
