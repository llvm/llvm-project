; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=xcore < %s | FileCheck %s

; CHECK: declare void @fiprintf(...)
; CHECK: declare void @iprintf(...)
; CHECK: declare void @siprintf(...)
