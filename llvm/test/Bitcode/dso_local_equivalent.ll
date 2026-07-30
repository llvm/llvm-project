; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder < %s

declare void @extern_func()

define void @call_extern_func() {
; CHECK: call void dso_local_equivalent @extern_func()
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

; CHECK: call void dso_local_equivalent @hidden_func()
define void @call_hidden_func() {
  call void dso_local_equivalent @hidden_func()
  ret void
}

define void @call_protected_func() {
; CHECK: call void dso_local_equivalent @protected_func()
  call void dso_local_equivalent @protected_func()
  ret void
}

define void @call_dso_local_func() {
; CHECK: call void dso_local_equivalent @dso_local_func()
  call void dso_local_equivalent @dso_local_func()
  ret void
}

define void @call_internal_func() {
; CHECK: call void dso_local_equivalent @internal_func()
  call void dso_local_equivalent @internal_func()
  ret void
}

define void @aliasee_func() {
entry:
  ret void
}

@alias_func = alias void (), void ()* @aliasee_func
@dso_local_alias_func = dso_local alias void (), void ()* @aliasee_func

define void @call_alias_func() {
; CHECK: call void dso_local_equivalent @alias_func()
  call void dso_local_equivalent @alias_func()
  ret void
}

define void @call_dso_local_alias_func() {
; CHECK: call void dso_local_equivalent @dso_local_alias_func()
  call void dso_local_equivalent @dso_local_alias_func()
  ret void
}

@ifunc_func = ifunc void (), void ()* ()* @resolver
@dso_local_ifunc_func = dso_local ifunc void (), void ()* ()* @resolver

define internal void ()* @resolver() {
entry:
  ret void ()* null
}

define void @call_ifunc_func() {
; CHECK: call void dso_local_equivalent @ifunc_func()
  call void dso_local_equivalent @ifunc_func()
  ret void
}

define void @call_dso_local_ifunc_func() {
; CHECK: call void dso_local_equivalent @dso_local_ifunc_func()
  call void dso_local_equivalent @dso_local_ifunc_func()
  ret void
}
