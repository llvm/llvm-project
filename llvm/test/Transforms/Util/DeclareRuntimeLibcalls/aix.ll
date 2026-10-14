; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=powerpc64-ibm-aix < %s | FileCheck %s

; CHECK: declare ptr @vec_calloc(i64, i64)
; CHECK: declare void @vec_free(ptr)
; CHECK: declare ptr @vec_malloc(i64)
; CHECK: declare ptr @vec_realloc(ptr, i64)
