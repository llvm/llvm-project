; REQUIRES: x86-registered-target
; RUN: llc < %s %loadnewpmbye | FileCheck %s --check-prefix=CHECK-ASM
; RUN: llc < %s %loadnewpmbye -plugin-arg=Bye,-last-words | FileCheck %s --check-prefix=CHECK-ACTIVE
; RUN: not llc < %s %loadnewpmbye -plugin-arg=Bye,-last-words -filetype=obj 2>&1 | FileCheck %s --check-prefix=CHECK-ERR

; LTO tools do not support -plugin-arg yet, so load Bye early to register its options
; RUN: llvm-as %s -o %t.bc
; RUN: llvm-lto2 run %t.bc -o %t.inactive -r %t.bc,somefunk,plx -r %t.bc,junk,plx -filetype=asm %loadnewpmbye
; RUN: FileCheck %s --check-prefix=CHECK-ASM < %t.inactive.0
; RUN: llvm-lto2 run %t.bc -o %t.active -r %t.bc,somefunk,plx -r %t.bc,junk,plx -filetype=asm %loadbye %loadnewpmbye -last-words
; RUN: FileCheck %s --check-prefix=CHECK-ACTIVE < %t.active.0
; RUN: not llvm-lto2 run %t.bc -o %t.err -r %t.bc,somefunk,plx -r %t.bc,junk,plx -filetype=obj %loadbye %loadnewpmbye -last-words 2>&1 | FileCheck %s --check-prefix=CHECK-LTO2-ERR

; REQUIRES: plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; Plugins are currently broken on AIX, at least in the CI.
; XFAIL: target={{.*}}-aix{{.*}}
; CHECK-ASM: somefunk:
; CHECK-ACTIVE: CodeGen Bye
; CHECK-ERR: error: last words unsupported for binary output
; CHECK-LTO2-ERR: last words unsupported for binary output

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}

