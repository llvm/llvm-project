; Check that matching orphan functions to profiles by demangled basename
; happens in a deterministic order.
;
; The candidates are collected in a StringMap keyed on the basename, so walking
; that map visits them in hash order. The four basenames here are chosen so that
; the hash order (bar, qux, baz, foo) differs from the sorted order, which makes
; this test fail if the sort in matchFunctionsWithoutProfileByBasename() is
; dropped -- an ordinary build is enough, no reverse-iteration build needed.

; REQUIRES: asserts
; RUN: llvm-profdata merge --sample --extbinary %S/Inputs/stale-profile-basename-match-order.prof -o %t.prof
; RUN: opt < %s -passes=sample-profile -sample-profile-file=%t.prof --salvage-stale-profile --salvage-unused-profile -S --debug-only=sample-profile-matcher 2>&1 | FileCheck %s

; CHECK:      Direct basename match: _Z3barl (IR) -> _Z3bari (Profile) [basename: bar]
; CHECK-NEXT: Direct basename match: _Z3bazl (IR) -> _Z3bazi (Profile) [basename: baz]
; CHECK-NEXT: Direct basename match: _Z3fool (IR) -> _Z3fooi (Profile) [basename: foo]
; CHECK-NEXT: Direct basename match: _Z3quxl (IR) -> _Z3quxi (Profile) [basename: qux]
; CHECK-NEXT: Direct basename matching found 4 matches

define void @_Z3fool(i64 %l) {
  ret void
}

define void @_Z3barl(i64 %l) {
  ret void
}

define void @_Z3bazl(i64 %l) {
  ret void
}

define void @_Z3quxl(i64 %l) {
  ret void
}
