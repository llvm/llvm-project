; Check that we reject 64-bit mode on 32-bit only CPUs.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target
; CHECK-ERROR64: LLVM ERROR: 64-bit code requested on a subtarget that doesn't support it!

; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=i386 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=i486 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=i586 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium-mmx 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=i686 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentiumpro 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium3m 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium-m 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=pentium4m 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=yonah 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=prescott 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=lakemont 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k6 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k6-2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=k6-3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-tbird 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-4 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-xp 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=athlon-mp 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=geode 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=winchip-c6 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=winchip2 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c3 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR64
; RUN: not llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=c3-2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR

define void @foo() {
  ret void
}
