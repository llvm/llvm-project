; RUN: llc -mtriple=x86_64-linux-gnu -relocation-model=pic -data-sections -o - %s --asm-verbose=0 | FileCheck %s -check-prefixes=CHECK

; Just ensure that we can write to an object file without error.
; RUN: llc -filetype=obj -mtriple=x86_64-linux-gnu -relocation-model=pic -data-sections -o /dev/null %s

declare void @extern_func()

; CHECK: call_extern_func:
; CHECK:       callq extern_func@PLT
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

; CHECK: call_hidden_func:
; CHECK:   callq hidden_func{{$}}
define void @call_hidden_func() {
  call void dso_local_equivalent @hidden_func()
  ret void
}

; CHECK: call_protected_func:
; CHECK:   callq protected_func{{$}}
define void @call_protected_func() {
  call void dso_local_equivalent @protected_func()
  ret void
}

; CHECK: call_dso_local_func:
; CHECK:   callq dso_local_func{{$}}
define void @call_dso_local_func() {
  call void dso_local_equivalent @dso_local_func()
  ret void
}

; CHECK: call_internal_func:
; CHECK:   callq internal_func{{$}}
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

; CHECK: call_alias_func:
; CHECK:   callq alias_func@PLT
define void @call_alias_func() {
  call void dso_local_equivalent @alias_func()
  ret void
}

; CHECK: call_dso_local_alias_func:
; CHECK:   callq .Ldso_local_alias_func$local{{$}}
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

; If an ifunc is not dso_local already, then we should still emit a stub for it
; to ensure it will be dso_local.
; CHECK: call_ifunc_func:
; CHECK:   callq ifunc_func@PLT
define void @call_ifunc_func() {
  call void dso_local_equivalent @ifunc_func()
  ret void
}

; CHECK: call_dso_local_ifunc_func:
; CHECK:   callq dso_local_ifunc_func{{$}}
define void @call_dso_local_ifunc_func() {
  call void dso_local_equivalent @dso_local_ifunc_func()
  ret void
}
