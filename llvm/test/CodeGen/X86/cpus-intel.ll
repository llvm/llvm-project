; Test that the CPU names work.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target

; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=i386 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=i486 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=i586 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium-mmx 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=i686 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentiumpro 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium3 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium3m 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium-m 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium4 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=pentium4m 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=yonah 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=prescott 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=i686-unknown-unknown -mcpu=lakemont 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty

; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=nocona 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=core2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=penryn 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=nehalem 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=corei7 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=westmere 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=sandybridge 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=corei7-avx 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=ivybridge 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=core-avx-i 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=haswell 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=core-avx2 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=broadwell 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=skylake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=skx 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=cascadelake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=cooperlake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=cannonlake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=icelake-client 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=icelake-server 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=tigerlake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=sapphirerapids 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=alderlake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=atom 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bonnell 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=silvermont 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=slm 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=goldmont 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=goldmont-plus 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=tremont 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=knl 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=knm 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty

define void @foo() {
  ret void
}
