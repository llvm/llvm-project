; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=wasm64-unknown-emscripten < %s | FileCheck %s

; CHECK: declare i32 @__small_fprintf(ptr, ptr, ...)
; CHECK: declare i32 @__small_printf(ptr, ...)
; CHECK: declare i32 @__small_sprintf(ptr, ptr, ...)
