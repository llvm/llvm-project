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

;; PR57815
;; Ensure dso_local_equivalent works the exact same way as the previous
;; examples but with forward-referenced symbols.

; CHECK: call_forward_func:
; CHECK:   callq forward_extern_func@PLT
define void @call_forward_func() {
  call void dso_local_equivalent @forward_extern_func()
  ret void
}

; CHECK: call_forward_hidden_func:
; CHECK:   callq forward_hidden_func{{$}}
define void @call_forward_hidden_func() {
  call void dso_local_equivalent @forward_hidden_func()
  ret void
}

; CHECK: call_forward_protected_func:
; CHECK:   callq forward_protected_func{{$}}
define void @call_forward_protected_func() {
  call void dso_local_equivalent @forward_protected_func()
  ret void
}

; CHECK: call_forward_dso_local_func:
; CHECK:   callq forward_dso_local_func{{$}}
define void @call_forward_dso_local_func() {
  call void dso_local_equivalent @forward_dso_local_func()
  ret void
}

; CHECK: call_forward_internal_func:
; CHECK:   callq forward_internal_func{{$}}
define void @call_forward_internal_func() {
  call void dso_local_equivalent @forward_internal_func()
  ret void
}

declare hidden void @forward_hidden_func()
declare protected void @forward_protected_func()
declare dso_local void @forward_dso_local_func()
define internal void @forward_internal_func() {
entry:
  ret void
}
define private void @forward_private_func() {
entry:
  ret void
}

; CHECK: call_forward_alias_func:
; CHECK:   callq forward_alias_func@PLT
define void @call_forward_alias_func() {
  call void dso_local_equivalent @forward_alias_func()
  ret void
}

; CHECK: call_forward_dso_local_alias_func:
; CHECK:   callq .Lforward_dso_local_alias_func$local{{$}}
define void @call_forward_dso_local_alias_func() {
  call void dso_local_equivalent @forward_dso_local_alias_func()
  ret void
}

define void @forward_aliasee_func() {
entry:
  ret void
}

@forward_alias_func = alias void (), ptr @forward_aliasee_func
@forward_dso_local_alias_func = dso_local alias void (), ptr @forward_aliasee_func

; If an ifunc is not dso_local already, then we should still emit a stub for it
; to ensure it will be dso_local.
; CHECK: call_forward_ifunc_func:
; CHECK:   callq forward_ifunc_func@PLT
define void @call_forward_ifunc_func() {
  call void dso_local_equivalent @forward_ifunc_func()
  ret void
}

; CHECK: call_forward_dso_local_ifunc_func:
; CHECK:   callq forward_dso_local_ifunc_func{{$}}
define void @call_forward_dso_local_ifunc_func() {
  call void dso_local_equivalent @forward_dso_local_ifunc_func()
  ret void
}

@forward_ifunc_func = ifunc void (), ptr @resolver
@forward_dso_local_ifunc_func = dso_local ifunc void (), ptr @resolver

;; Test "no-named" variables
; CHECK: call_no_name_hidden:
; CHECK:   callq __unnamed_{{[0-9]+}}{{$}}
define void @call_no_name_hidden() {
  call void dso_local_equivalent @0()
  ret void
}

; CHECK: call_no_name_extern:
; CHECK:   callq __unnamed_{{[0-9]+}}@PLT
define void @call_no_name_extern() {
  call void dso_local_equivalent @1()
  ret void
}

declare hidden void @0()
declare void @1()

;; Note that we keep this at the very end because llc emits this after all the
;; functions.
; CHECK: const:
; CHECK:   .long   forward_extern_func@PLT
@const = constant i32 trunc (i64 ptrtoint (ptr dso_local_equivalent @forward_extern_func to i64) to i32)

declare void @forward_extern_func()
