; REQUIRES: webassembly-registered-target
; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=powerpc64-ibm-aix < %s | FileCheck %s

; CHECK: declare noalias noundef ptr @vec_calloc(i64 noundef, i64 noundef) [[VEC_CALLOC_ATTRS:#[0-9]+]]
; CHECK: declare void @vec_free(ptr allocptr noundef captures(none)) [[VEC_FREE_ATTRS:#[0-9]+]]
; CHECK: declare noalias noundef ptr @vec_malloc(i64 noundef) [[VEC_MALLOC_ATTRS:#[0-9]+]]
; CHECK: declare noalias noundef ptr @vec_realloc(ptr allocptr captures(none), i64 noundef) [[VEC_REALLOC_ATTRS:#[0-9]+]]

; CHECK: attributes [[VEC_CALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite, errnomem: write) "alloc-family"="vec_malloc" }
; CHECK: attributes [[VEC_FREE_ATTRS]] = { nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="vec_malloc" }
; CHECK: attributes [[VEC_MALLOC_ATTRS]] = { nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite, errnomem: write) "alloc-family"="vec_malloc" }
; CHECK: attributes [[VEC_REALLOC_ATTRS]] = { nounwind willreturn allockind("realloc") allocsize(1) memory(argmem: readwrite, inaccessiblemem: readwrite, errnomem: write) "alloc-family"="vec_malloc" }
