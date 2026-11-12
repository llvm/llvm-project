; Check that we reject 64-bit mode on 32-bit only CPUs.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target
; CHECK-ERROR64: LLVM ERROR: 64-bit code requested on a subtarget that doesn't support it!

; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k6 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k6-2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k6-3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-tbird 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-xp 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-mp 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=geode 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64

define void @foo() {
  ret void
}
