; RUN: opt -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass=Structure -O2 %s -enable-new-pm=0 2>&1 | FileCheck -check-prefix=LPM %s
; RUN: opt -mtriple=amdgcn--amdhsa -disable-output -disable-verify -debug-pass-manager -passes='default<O2>' %s 2>&1 | FileCheck -check-prefix=NPM %s

; LPM: Function Integration/Inlining
; LPM: FunctionPass Manager
; LPM: Infer address spaces
; LPM: SROA

; NPM: Running pass: InlinerPass
; NPM: Running pass: InferAddressSpacesPass
; NPM: Running pass: SROA

define void @empty() {
  ret void
}
