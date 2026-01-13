; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=wasm64-unknown-emscripten < %s | FileCheck %s

; CHECK: declare void @__small_fprintf(...)
; CHECK: declare void @__small_printf(...)
; CHECK: declare void @__small_sprintf(...)
