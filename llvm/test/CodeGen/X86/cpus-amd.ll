; Test that the CPU names work.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target

; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=k6 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=k6-2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=k6-3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=athlon 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=athlon-tbird 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=athlon-4 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=athlon-xp 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=athlon-mp 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty

; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k8 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=opteron 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon64 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-fx 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k8-sse3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=opteron-sse3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon64-sse3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=amdfam10 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=barcelona 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver1 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bdver4 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=btver1 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=btver2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=znver1 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=znver2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=znver3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=znver4 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty

define void @foo() {
  ret void
}
