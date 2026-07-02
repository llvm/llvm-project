; RUN: opt < %s -passes='cg-profile' -S | FileCheck %s

; A function whose entire name is a strippable suffix (e.g. ".llvm.123")
; canonicalizes to an empty name. Building the InstrProfSymtab
; for such a name returns an error that cg-profile intentionally ignores.

; CHECK: define void @.llvm.123()
define void @.llvm.123() {
  ret void
}
