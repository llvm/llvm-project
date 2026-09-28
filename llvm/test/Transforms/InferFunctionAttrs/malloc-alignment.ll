; RUN: opt < %s -mtriple=aarch64-unknown-linux-gnu -passes=inferattrs -S | FileCheck %s --match-full-lines --check-prefix=STRONG
; RUN: opt < %s -mtriple=aarch64-unknown-unknown -passes=inferattrs -S | FileCheck %s --match-full-lines --check-prefix=NONE

; Strong alignment: the C library guarantees that malloc and calloc return
; pointers aligned to alignof(max_align_t) for every size.
; No guarantee: no C library is assumed, so none is assumed either (C23 /
; WG14 N2293 allows weaker).
;
; More targets can be added to the STRONG runs here as their C library's
; malloc/calloc alignment guarantee is confirmed (together with an entry in
; hasStrongMallocAlignment in TargetLibraryInfo.cpp).

; STRONG: declare noalias noundef ptr @malloc(i64 noundef) #{{[0-9]+}}
; NONE: declare noalias noundef ptr @malloc(i64 noundef) #{{[0-9]+}}
declare ptr @malloc(i64)

; STRONG: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) #{{[0-9]+}}
; NONE: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) #{{[0-9]+}}
declare ptr @calloc(i64, i64)