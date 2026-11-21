; RUN: not llvm-split -enable-call-graph-split-module=true -j2 -o %t %s 2>&1 | FileCheck %s

; Test the currently unsupported case of a GlobalAlias whose aliasee is
; assigned to a different partition. Aliases are always placed in partition 0,
; where the aliasee then only appears as a declaration, which the verifier
; rejects ("Alias must point to a definition").
;
; TODO: A follow-up patch will keep an alias and its aliasee in the same
; partition. Update this test to check the fixed behavior at that point.

; CHECK: Alias must point to a definition

@alias_fn = alias void (), ptr @aliasee

; @aliasee is an expensive call-graph root, so the cost-based partitioning
; places it in partition 1, away from its alias in partition 0.
define void @aliasee() {
  call void @aliasee_helper()
  ret void
}

define void @aliasee_helper() {
  ret void
}

define void @root() {
  ret void
}
