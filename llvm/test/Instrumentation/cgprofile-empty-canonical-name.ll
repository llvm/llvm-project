; RUN: opt < %s -passes='cg-profile' -S | FileCheck %s

; A function whose entire name is a strippable suffix (e.g. ".llvm.123")
; canonicalizes to an empty name.

; CHECK: define void @.llvm.123()
define void @.llvm.123() {
  ret void
}
