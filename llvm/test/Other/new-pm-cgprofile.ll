; RUN: opt -debug-pass-manager -passes='default<O2>' %s 2>&1 |FileCheck %s --check-prefixes=DEFAULT
; RUN: opt -debug-pass-manager -passes='default<O2>' -enable-npm-call-graph-profile=0 %s 2>&1 |FileCheck %s --check-prefixes=OFF
; RUN: opt -debug-pass-manager -passes='default<O2>' -enable-npm-call-graph-profile=1 %s 2>&1 |FileCheck %s --check-prefixes=ON
;
; DEFAULT: Running pass: CGProfilePass
; OFF-NOT: Running pass: CGProfilePass
; ON: Running pass: CGProfilePass

define void @foo() {
  ret void
}
