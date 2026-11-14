; The input is intentionally invalid IR, so
; -disable-verify is needed to get it past opt's implicit input verification; the
; trailing explicit "verify" pass then proves globalopt produced valid IR.
; RUN: opt -mtriple=x86_64-pc-windows-msvc -disable-verify -passes=globalopt,verify -S < %s | FileCheck %s
; RUN: opt -mtriple=x86_64-unknown-linux-gnu -disable-verify -passes=globalopt,verify -S < %s | FileCheck %s

$foo = comdat any

; @foo is the comdat leader (name == comdat). It is available_externally and has
; a live comdat member (@bar), so it is preserved -- and promoted off
; available_externally to weak_odr so it stays a valid, emitted comdat key.
; CHECK: define weak_odr void @foo() local_unnamed_addr comdat {
define available_externally void @foo() comdat($foo) {
  ret void
}

; @bar is a member of comdat $foo, kept alive by @keep.
; CHECK: define void @bar() local_unnamed_addr comdat($foo) {
define void @bar() comdat($foo) {
  ret void
}

; CHECK: define void @keep() local_unnamed_addr {
define void @keep() {
  call void @bar()
  ret void
}
