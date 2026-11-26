; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=wasm64-unknown-emscripten < %s | FileCheck %s

; CHECK: declare noundef i32 @__small_fprintf(ptr noundef captures(none), ptr noundef readonly captures(none), ...) [[PRINTF_ATTRS:#[0-9]+]]
; CHECK: declare noundef i32 @__small_printf(ptr noundef readonly captures(none), ...) [[PRINTF_ATTRS]]
; CHECK: declare noundef i32 @__small_sprintf(ptr noalias noundef writeonly captures(none), ptr noundef readonly captures(none), ...) [[PRINTF_ATTRS]]
; CHECK: declare float @logf(float) [[LOGF_ATTRS:#[0-9]+]]

; CHECK: attributes [[LOGF_ATTRS]] = { nocallback nofree nosync nounwind willreturn memory(none) }
; CHECK: attributes [[PRINTF_ATTRS]] = { nofree nounwind }
