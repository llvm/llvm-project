; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mattr=+aix-shared-lib-tls-model-opt \
; RUN:   -ppc-asm-full-reg-names < %s | FileCheck %s
; RUN: not llc -mtriple powerpc-ibm-aix-xcoff -mattr=+aix-shared-lib-tls-model-opt \
; RUN:   -ppc-asm-full-reg-names < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK-NOT-SUPPORTED
; RUN: not llc -mtriple powerpc64le-unknown-linux-gnu -mattr=+aix-shared-lib-tls-model-opt \
; RUN:   -ppc-asm-full-reg-names < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=CHECK-NOT-SUPPORTED

define dso_local signext i32 @testNoIRAttr() {
entry:
  ret i32 0
}

; Check that the aix-shared-lib-tls-model-opt attribute is not supported on Linux and AIX (32-bit).
; CHECK-NOT-SUPPORTED: The aix-shared-lib-tls-model-opt attribute is only supported on AIX in 64-bit mode.

; Make sure that the test was actually compiled successfully after using the
; aix-shared-lib-tls-model-opt attribute.
; CHECK-LABEL: testNoIRAttr:
; CHECK:        li r3, 0
; CHECK-NEXT:   blr
