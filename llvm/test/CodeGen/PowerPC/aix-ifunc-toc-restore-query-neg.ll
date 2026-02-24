; REQUIRES: target=powerpc{{.*}}
; REQUIRES: asserts
; This testcase is for testing the negative return values of the
; TOCRestoreNeededForCallToImplementation query.

; RUN: llc < %s -mtriple=powerpc64-ibm-aix-xcoff -test-ifunc-warn-noerror -filetype=obj -o /dev/null 2>&1 | FileCheck %s

; CHECK: TOC register save/restore needed for ifunc "foo_ext_decl_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_default_decl_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_weak_decl_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_hidden_weak_decl_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_protected_weak_decl_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_default_weak_decl_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_def_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_default_def_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_weak_def_ifunc"
; CHECK: TOC register save/restore needed for ifunc "foo_ext_default_weak_def_ifunc"


define void @foo_ext_def() {
entry:
  ret void
}
define void @foo_ext_default_def() {
entry:
  ret void
}
define weak void @foo_ext_default_weak_def() {
entry:
  ret void
}
define weak void @foo_ext_weak_def() {
entry:
  ret void
}
declare void @foo_ext_decl(...) 
declare void @foo_ext_default_decl(...) 
declare extern_weak void @foo_ext_weak_decl(...) 
declare extern_weak hidden void @foo_ext_hidden_weak_decl(...) 
declare extern_weak protected void @foo_ext_protected_weak_decl(...) 
declare extern_weak void @foo_ext_default_weak_decl(...) 


define internal ptr @foo_ext_decl_resolver() {
entry:
  ret ptr @foo_ext_decl
}
define internal ptr @foo_ext_default_decl_resolver() {
entry:
  ret ptr @foo_ext_default_decl
}
define internal ptr @foo_ext_weak_decl_resolver() {
entry:
  ret ptr @foo_ext_weak_decl
}
define internal ptr @foo_ext_hidden_weak_decl_resolver() {
entry:
  ret ptr @foo_ext_hidden_weak_decl
}
define internal ptr @foo_ext_protected_weak_decl_resolver() {
entry:
  ret ptr @foo_ext_protected_weak_decl
}
define internal ptr @foo_ext_default_weak_decl_resolver() {
entry:
  ret ptr @foo_ext_default_weak_decl
}
define internal ptr @foo_ext_def_resolver() {
entry:
  ret ptr @foo_ext_def
}
define internal ptr @foo_ext_default_def_resolver() {
entry:
  ret ptr @foo_ext_default_def
}
define internal ptr @foo_ext_weak_def_resolver() {
entry:
  ret ptr @foo_ext_weak_def
}
define internal ptr @foo_ext_default_weak_def_resolver() {
entry:
  ret ptr @foo_ext_default_weak_def
}

@foo_ext_decl_ifunc = ifunc i32 (...), ptr @foo_ext_decl_resolver
@foo_ext_default_decl_ifunc = ifunc i32 (...), ptr @foo_ext_default_decl_resolver
@foo_ext_weak_decl_ifunc = ifunc i32 (...), ptr @foo_ext_weak_decl_resolver
@foo_ext_hidden_weak_decl_ifunc = ifunc i32 (...), ptr @foo_ext_hidden_weak_decl_resolver
@foo_ext_protected_weak_decl_ifunc = ifunc i32 (...), ptr @foo_ext_protected_weak_decl_resolver
@foo_ext_default_weak_decl_ifunc = ifunc i32 (...), ptr @foo_ext_default_weak_decl_resolver
@foo_ext_def_ifunc = ifunc i32 (...), ptr @foo_ext_def_resolver
@foo_ext_default_def_ifunc = ifunc i32 (...), ptr @foo_ext_default_def_resolver
@foo_ext_weak_def_ifunc = ifunc i32 (...), ptr @foo_ext_weak_def_resolver
@foo_ext_default_weak_def_ifunc = ifunc i32 (...), ptr @foo_ext_default_weak_def_resolver


