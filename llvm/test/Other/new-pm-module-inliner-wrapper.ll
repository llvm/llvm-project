; RUN: opt -disable-verify -debug-pass-manager -passes='inliner-wrapper' -S %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ALWAYS
; RUN: opt -disable-verify -disable-always-inliner-in-module-wrapper -debug-pass-manager -passes='inliner-wrapper' -S %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DISABLEALWAYS

; DISABLEALWAYS-NOT: Running pass: AlwaysInlinerPass
; ALWAYS: Running pass: AlwaysInlinerPass
; CHECK: Running pass: InlinerPass

define void @foo() {
  ret void
}
