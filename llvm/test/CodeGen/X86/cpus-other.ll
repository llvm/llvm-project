; Test that the CPU names work.
;
; First ensure the error message matches what we expect.
; CHECK-ERROR: not a recognized processor for this target

; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=foobar 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; Now ensure the error message doesn't occur for valid CPUs.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target

; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=generic 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty

; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=geode 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=winchip-c6 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=winchip2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=c3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=c3-2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty

;; x86-64 micro-architecture levels.
; RUN: llc %s -filetype=null -mtriple=x86_64 -mcpu=x86-64-v2
; RUN: llc %s -filetype=null -mtriple=x86_64 -mcpu=x86-64-v3
; RUN: llc %s -filetype=null -mtriple=x86_64 -mcpu=x86-64-v4

define void @foo() {
  ret void
}
