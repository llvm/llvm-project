; RUN: not --crash opt < %s -wasm-lower-em-ehsjlj -wasm-enable-sjlj -S 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

; CHECK: LLVM ERROR: Function setjmp_longjmp is using setjmp/longjmp but does not have +exception-handling target feature
define void @setjmp_longjmp() {
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %call = call i32 @setjmp(ptr %buf) #0
  call void @longjmp(ptr %buf, i32 1) #1
  unreachable
}

; Function Attrs: returns_twice
declare i32 @setjmp(ptr) #0
; Function Attrs: noreturn
declare void @longjmp(ptr, i32) #1

attributes #0 = { returns_twice }
attributes #1 = { noreturn }
