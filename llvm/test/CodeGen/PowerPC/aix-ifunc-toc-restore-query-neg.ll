; This testcase is for testing the negative return values of the
; TOCRestoreNeededForCallToImplementation query.

; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff -test-ifunc-warn-noerror -filetype=obj -o /dev/null 2>&1 | FileCheck %s

; CHECK: TOC register save/restore needed for ifunc "foo_extern"
; CHECK: TOC register save/restore needed for ifunc "foo_decl"
; CHECK: TOC register save/restore needed for ifunc "foo_weak_hidden"
; CHECK: TOC register save/restore needed for ifunc "foo_weak_protected"
; CHECK: TOC register save/restore needed for ifunc "foo_weak_extern"

@foo_extern = ifunc i32 (...), ptr @resolve_extern
@foo_decl = ifunc i32 (...), ptr @resolve_decl
@foo_weak_hidden = ifunc i32 (...), ptr @resolve_weak_hidden
@foo_weak_protected = ifunc i32 (...), ptr @resolve_weak_protected
@foo_weak_extern = ifunc i32 (...), ptr @resolve_weak_extern

define i32 @bar_extern() {
entry:
  ret i32 5
}
define internal nonnull ptr @resolve_extern() {
entry:
  ret ptr @bar_extern
}

declare i32 @bar_decl()
define internal nonnull ptr @resolve_decl() {
entry:
  ret ptr @bar_decl
}

define weak hidden i32 @bar_weak_hidden() {
entry:
  ret i32 2
}
define internal nonnull ptr @resolve_weak_hidden() {
entry:
  ret ptr @bar_weak_hidden
}

define weak protected i32 @bar_weak_protected() {
entry:
  ret i32 3
}
define internal nonnull ptr @resolve_weak_protected() {
entry:
  ret ptr @bar_weak_protected
}

define weak i32 @bar_weak_extern() {
entry:
  ret i32 5
}
define internal nonnull ptr @resolve_weak_extern() {
entry:
  ret ptr @bar_weak_extern
}

