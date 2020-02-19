; Check that we reject 64-bit mode on 32-bit only CPUs.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target
; CHECK-ERROR64: LLVM ERROR: 64-bit code requested on a subtarget that doesn't support it!

; RUN: not --crash llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=winchip-c6 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not --crash llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=winchip2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not --crash llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not --crash llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c3-2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR

define void @foo() {
  ret void
}
