; RUN: opt -S < %s | FileCheck %s --check-prefixes=ALL,FIRST
; RUN: opt -S -globalopt < %s | FileCheck %s --check-prefixes=ALL,SECOND
;
; Make sure we use FIRST to check for @sometimes_here as ALL does not work.

define internal void @sometimes_here() {
  ret void
}

define void @always_here() {
  ret void
}
