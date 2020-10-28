; RUN: llc < %s -asm-verbose=false -enable-emscripten-sjlj -wasm-keep-registers | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

declare i32 @setjmp(%struct.__jmp_buf_tag*) #0
declare {i32, i32} @returns_struct()

; Test the combination of backend legalization of large return types and the
; Emscripten sjlj transformation
define {i32, i32} @legalized_to_sret() {
entry:
  %env = alloca [1 x %struct.__jmp_buf_tag], align 16
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %env, i32 0, i32 0
  %call = call i32 @setjmp(%struct.__jmp_buf_tag* %arraydecay) #0
  ; This is the function pointer to pass to invoke.
  ; It needs to be the first argument (that's what we're testing here)
  ; CHECK: i32.const $push[[FPTR:[0-9]+]]=, returns_struct
  ; This is the sret stack region (as an offset from the stack pointer local)
  ; CHECK: call invoke_vi, $pop[[FPTR]]
  %ret = call {i32, i32} @returns_struct()
  ret {i32, i32} %ret
}

attributes #0 = { returns_twice }
