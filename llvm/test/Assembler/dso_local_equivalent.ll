; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

declare void @extern_func()

; CHECK: @call_extern_func
; CHECK: call void dso_local_equivalent @extern_func()
define void @call_extern_func() {
  call void dso_local_equivalent @extern_func()
  ret void
}

declare hidden void @hidden_func()
declare protected void @protected_func()
declare dso_local void @dso_local_func()
define internal void @internal_func() {
entry:
  ret void
}
define private void @private_func() {
entry:
  ret void
}

; CHECK: @call_hidden_func
; CHECK: call void dso_local_equivalent @hidden_func()
define void @call_hidden_func() {
  call void dso_local_equivalent @hidden_func()
  ret void
}

; CHECK: @call_protected_func
; CHECK: call void dso_local_equivalent @protected_func()
define void @call_protected_func() {
  call void dso_local_equivalent @protected_func()
  ret void
}

; CHECK: @call_dso_local_func
; CHECK: call void dso_local_equivalent @dso_local_func()
define void @call_dso_local_func() {
  call void dso_local_equivalent @dso_local_func()
  ret void
}

; CHECK: @call_internal_func
; CHECK: call void dso_local_equivalent @internal_func()
define void @call_internal_func() {
  call void dso_local_equivalent @internal_func()
  ret void
}

define void @aliasee_func() {
entry:
  ret void
}

@alias_func = alias void (), ptr @aliasee_func
@dso_local_alias_func = dso_local alias void (), ptr @aliasee_func

; CHECK: @call_alias_func
; CHECK: call void dso_local_equivalent @alias_func()
define void @call_alias_func() {
  call void dso_local_equivalent @alias_func()
  ret void
}

; CHECK: @call_dso_local_alias_func
; CHECK: call void dso_local_equivalent @dso_local_alias_func()
define void @call_dso_local_alias_func() {
  call void dso_local_equivalent @dso_local_alias_func()
  ret void
}

@ifunc_func = ifunc void (), ptr @resolver
@dso_local_ifunc_func = dso_local ifunc void (), ptr @resolver

define internal ptr @resolver() {
entry:
  ret ptr null
}

; CHECK: @call_ifunc_func
; CHECK: call void dso_local_equivalent @ifunc_func()
define void @call_ifunc_func() {
  call void dso_local_equivalent @ifunc_func()
  ret void
}

; CHECK: @call_dso_local_ifunc_func
; CHECK: call void dso_local_equivalent @dso_local_ifunc_func()
define void @call_dso_local_ifunc_func() {
  call void dso_local_equivalent @dso_local_ifunc_func()
  ret void
}
