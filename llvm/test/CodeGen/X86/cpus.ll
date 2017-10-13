; Test that the CPU names work.
;
; First ensure the error message matches what we expect.
; CHECK-ERROR: not a recognized processor for this target
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=foobar 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR
;
; Now ensure the error message doesn't occur for valid CPUs.
; CHECK-NO-ERROR-NOT: not a recognized processor for this target
;
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
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=broadwell 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=cannonlake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=skylake 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=skylake-avx512 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=skx 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=atom 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=bonnell 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=silvermont 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=slm 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=goldmont 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=lakemont 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
; RUN: llc < %s -o /dev/null -mtriple=x86_64-unknown-unknown -mcpu=knl 2>&1 | FileCheck %s --check-prefix=CHECK-NO-ERROR --allow-empty
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

