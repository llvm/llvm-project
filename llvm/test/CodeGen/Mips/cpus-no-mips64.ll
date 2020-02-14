; Check that we reject 64-bit mode on 32-bit only CPUs.

; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips1 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips2 2>&1 | FileCheck %s

; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips32 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips32r2 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips32r3 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips32r5 2>&1 | FileCheck %s
; RUN: not --crash llc < %s -o /dev/null -mtriple=mips64 -mcpu=mips32r6 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: 64-bit code requested on a subtarget that doesn't support it!

define void @foo() {
  ret void
}
