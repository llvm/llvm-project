; RUN: not --crash llc < %s -enable-emscripten-sjlj 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -wasm-enable-sjlj -mattr=+exception-handling -exception-model=wasm 2>&1 | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

%struct.__jmp_buf_tag = type { [6 x i32], i32, [32 x i32] }

; CHECK: LLVM ERROR: Indirect use of setjmp is not supported
@setjmp_fp = global i32 (%struct.__jmp_buf_tag*)* @setjmp, align 4

define void @indirect_setjmp_call() {
entry:
  %buf = alloca [1 x %struct.__jmp_buf_tag], align 16
  %0 = load i32 (%struct.__jmp_buf_tag*)*, i32 (%struct.__jmp_buf_tag*)** @setjmp_fp, align 4
  %arraydecay = getelementptr inbounds [1 x %struct.__jmp_buf_tag], [1 x %struct.__jmp_buf_tag]* %buf, i32 0, i32 0
  %call = call i32 %0(%struct.__jmp_buf_tag* %arraydecay)
  call void @foo()
  ret void
}

declare void @foo()
; Function Attrs: returns_twice
declare i32 @setjmp(%struct.__jmp_buf_tag*) #0

attributes #0 = { returns_twice }

