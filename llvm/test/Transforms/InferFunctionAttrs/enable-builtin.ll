; RUN: opt -S -mtriple=amdgcn-- -passes=inferattrs %s | FileCheck -check-prefix=NOBUILTIN %s
; RUN: opt -S -enable-builtin=malloc -mtriple=amdgcn-- -passes=inferattrs %s | FileCheck -check-prefix=WITHBUILTIN %s

; Test that the -enable-builtin flag works and forces recognition of
; malloc despite the target's default TargetLibraryInfo.

; NOBUILTIN: declare ptr @malloc(i64)

; WITHBUILTIN: declare noalias noundef ptr @malloc(i64 noundef) #0
; WITHBUILTIN: attributes #0 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" }

declare ptr @malloc(i64)

