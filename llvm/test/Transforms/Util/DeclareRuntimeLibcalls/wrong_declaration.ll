; RUN: opt -S -passes=declare-runtime-libcalls -mtriple=x86_64-apple-macos10.9 < %s | FileCheck %s

; Make sure there is no crash if there are definitions or declarations
; with the wrong type signature.

; CHECK: define void @sqrtf() {
define void @sqrtf() {
  ret void
}

; CHECK: define float @sqrt(float %0) {
define float @sqrt(float) {
  ret float 0.0
}

; CHECK: declare double @__sincos_stret(double)
declare double @__sincos_stret(double)

; CHECK: declare { float, float } @__sincosf_stret(float)
declare { float, float } @__sincosf_stret(float)

