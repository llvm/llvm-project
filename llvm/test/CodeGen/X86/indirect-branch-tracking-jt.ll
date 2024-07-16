; RUN: sed 's/level/0/g' %s | llc -o - -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefixes=ALL,LEVEL0
; RUN: sed 's/level/1/g' %s | llc -o - -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefixes=ALL,LEVEL1
; RUN: sed 's/level/2/g' %s | llc -o - -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefixes=ALL,LEVEL2
; Check the following for IBT protection of jump tables:
;  - if cf-protection-branch == 0, then no ENDBRANCH instructions are inserted and the indirect jump table branch does not have a NOTRACK prefix.
;  - if cf-protection-branch == 1, then an ENDBRANCH is inserted at function entry but *not* in any jump table BB, and the indirect jump table branch *has* a NOTRACK prefix.
;  - if cf-protection-branch >= 2, then an ENDBRANCH is inserted at function antry *and* in the jump table BBs, and the indirect jump table branch *does not have* a NOTRACK prefix.

define void @foo(i32 %x) {
; ALL-LABEL: foo
; LEVEL0-NOT: endbr64
; LEVEL1: endbr64
; LEVEL2: endbr64
; LEVEL0: jmpq *
; LEVEL1: notrack jmpq *
; LEVEL2: jmpq *
; ALL-LABEL: .LBB0_2:
; LEVEL0-NOT: endbr64
; LEVEL1-NOT: endbr64
; LEVEL2: endbr64
; ALL: retq
  switch i32 %x, label %sw.default [
    i32 0, label %sw.bb
    i32 1, label %sw.bb1
    i32 2, label %sw.bb2
    i32 3, label %sw.bb3
  ]

sw.bb:
  ret void

sw.bb1:
  ret void

sw.bb2:
  ret void

sw.bb3:
  ret void

sw.default:
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 8, !"cf-protection-branch", i32 level}
