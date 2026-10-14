; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=xcore < %s | FileCheck %s

; CHECK: declare i32 @fiprintf(ptr, ptr, ...)
; CHECK: declare i32 @iprintf(ptr, ...)
; CHECK: declare i32 @siprintf(ptr, ptr, ...)
