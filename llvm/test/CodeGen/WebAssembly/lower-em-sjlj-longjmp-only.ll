; RUN: opt < %s -wasm-lower-em-ehsjlj -S | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [32 x i32] }

@buffer = global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 16

; Tests if program does not crash when there's no setjmp function calls in the
; module.

; CHECK: call void @emscripten_longjmp_jmpbuf
define void @longjmp_only() {
entry:
  call void @longjmp(%struct.__jmp_buf_tag* getelementptr inbounds ([1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* @buffer, i32 0, i32 0), i32 1) #1
  unreachable
}

; Function Attrs: noreturn nounwind
declare void @longjmp(%struct.__jmp_buf_tag*, i32) #1

attributes #1 = { noreturn nounwind }
