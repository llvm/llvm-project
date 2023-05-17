; RUN: opt -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass-manager -passes='default<O2>' %s 2>&1 | FileCheck -check-prefix=NPM %s

; NPM: Running pass: InlinerPass
; NPM: Running pass: InferAddressSpacesPass
; NPM: Running pass: SROA

define void @empty() {
  ret void
}
