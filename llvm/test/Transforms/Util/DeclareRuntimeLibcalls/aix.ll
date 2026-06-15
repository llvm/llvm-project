; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=powerpc64-ibm-aix < %s | FileCheck %s

; CHECK: declare void @vec_calloc(...)
; CHECK: declare void @vec_free(...)
; CHECK: declare void @vec_malloc(...)
; CHECK: declare void @vec_realloc(...)
