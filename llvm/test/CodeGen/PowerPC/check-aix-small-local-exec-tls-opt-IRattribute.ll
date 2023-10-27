; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -ppc-asm-full-reg-names \
; RUN:   < %s | FileCheck %s
; RUN: not llc -mtriple powerpc-ibm-aix-xcoff -ppc-asm-full-reg-names \
; RUN:   < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-SUPPORTED
; RUN: not llc -mtriple powerpc64le-unknown-linux-gnu -ppc-asm-full-reg-names \
; RUN:   < %s 2>&1 | FileCheck %s --check-prefix=CHECK-NOT-SUPPORTED

define dso_local signext i32 @testWithIRAttr() #0 {
entry:
  ret i32 0
}
; Check that the aix-small-local-exec-tls attribute is not supported on Linux and AIX (32-bit).
; CHECK-NOT-SUPPORTED: The aix-small-local-exec-tls attribute is only supported on AIX in 64-bit mode.

; Make sure that the test was actually compiled successfully after using the
; aix-small-local-exec-tls attribute.
; CHECK-LABEL: testWithIRAttr:
; CHECK:        li r3, 0
; CHECK-NEXT:   blr


attributes #0 = { "target-features"="+aix-small-local-exec-tls" }

