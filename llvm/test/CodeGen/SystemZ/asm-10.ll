; Test the FPR constraint "f".
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -no-integrated-as | FileCheck %s

define float @f1() {
; CHECK-LABEL: f1:
; CHECK: lzer %f1
; CHECK: blah %f0 %f1
; CHECK: br %r14
  %val = call float asm "blah $0 $1", "=&f,f" (float 0.0)
  ret float %val
}

define double @f2() {
; CHECK-LABEL: f2:
; CHECK: lzdr %f1
; CHECK: blah %f0 %f1
; CHECK: br %r14
  %val = call double asm "blah $0 $1", "=&f,f" (double 0.0)
  ret double %val
}

define double @f3() {
; CHECK-LABEL: f3:
; CHECK: lzxr %f1
; CHECK: blah %f0 %f1
; CHECK: br %r14
  %val = call double asm "blah $0 $1", "=&f,f" (fp128 0xL00000000000000000000000000000000)
  ret double %val
}
