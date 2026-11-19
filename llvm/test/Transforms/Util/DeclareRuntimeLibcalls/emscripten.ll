; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=wasm64-unknown-emscripten < %s | FileCheck %s

; CHECK: declare i32 @__small_fprintf(ptr, ptr, ...)
; CHECK: declare i32 @__small_printf(ptr, ...)
; CHECK: declare i32 @__small_sprintf(ptr, ptr, ...)
; CHECK: declare float @logf(float) [[LOGF_ATTRS:#[0-9]+]]

; CHECK: attributes [[LOGF_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
