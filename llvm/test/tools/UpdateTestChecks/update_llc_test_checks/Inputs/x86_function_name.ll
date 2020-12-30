;; Check that we accept functions with '$' in the name.
; RUN: llc -mtriple=x86_64 < %s | FileCheck %s

;; Check that we accept .Ldsolocal$local: below the function label.
; RUN: llc -mtriple=x86_64 -relocation-model=pic < %s | FileCheck %s --check-prefix=PIC

;; Check that we accept .seh_proc below the function label.
; RUN: llc -mtriple=x86_64-windows -relocation-model=pic < %s | FileCheck %s --check-prefix=WIN

define hidden i32 @"_Z54bar$ompvariant$bar"() {
entry:
  ret i32 2
}

define dso_local i32 @dsolocal() {
entry:
  call void @ext()
  ret i32 2
}

declare void @ext()
