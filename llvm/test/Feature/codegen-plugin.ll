; REQUIRES: x86-registered-target
; RUN: llc < %s %loadnewpmbye | FileCheck %s --check-prefix=CHECK-ASM
; RUN: llc < %s %loadnewpmbye -last-words | FileCheck %s --check-prefix=CHECK-ACTIVE
; RUN: not llc < %s %loadnewpmbye -last-words -filetype=obj 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
; REQUIRES: plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; Plugins are currently broken on AIX, at least in the CI.
; XFAIL: target={{.*}}-aix{{.*}}
; CHECK-ASM: somefunk:
; CHECK-ACTIVE: CodeGen Bye
; CHECK-ERR: error: last words unsupported for binary output

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}

