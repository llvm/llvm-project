; REQUIRES: asserts

; Test pass name: ppc-ctr-loops-verify.
; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-before=ppc-ctr-loops-verify -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-BEFORE-CTR-LOOPS-VERIFY
; STOP-BEFORE-CTR-LOOPS-VERIFY-NOT: -ppc-ctr-loops-verify
; STOP-BEFORE-CTR-LOOPS-VERIFY-NOT: "ppc-ctr-loops-verify" pass is not registered.
; STOP-BEFORE-CTR-LOOPS-VERIFY-NOT: PowerPC CTR Loops Verify

; RUN: llc -mtriple=powerpc64le-unknown-unknown < %s -debug-pass=Structure -stop-after=ppc-ctr-loops-verify -o /dev/null 2>&1 | FileCheck %s -check-prefix=STOP-AFTER-CTR-LOOPS-VERIFY
; STOP-AFTER-CTR-LOOPS-VERIFY: -ppc-ctr-loops-verify
; STOP-AFTER-CTR-LOOPS-VERIFY-NOT: "ppc-ctr-loops-verify" pass is not registered.
; STOP-AFTER-CTR-LOOPS-VERIFY: PowerPC CTR Loops Verify
