target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

declare void @dead_func()

; Called from a @dead_func() in the other file, should not be imported there
; Ensure the cycle formed by calling @dead_func doesn't prevent stripping.
define void @baz() {
    call void @dead_func()
    ret void
}

; Called via llvm.global_ctors, should be detected as live via the
; marking of llvm.global_ctors as a live root in the index.
define void @boo() {
  ret void
}

define void @another_dead_func() {
    call void @dead_func()
    ret void
}

define linkonce_odr void @linkonceodrfuncwithalias() {
entry:
  ret void
}
