; Test that the CPU names work.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target

; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c86-4g-m4 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c86-4g-m6 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c86-4g-m7 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c86-4g-m8 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty

define void @foo() {
  ret void
}
