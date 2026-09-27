; REQUIRES: x86-registered-target
; RUN: opt %s %loadnewpmbye %loadbye -passes="goodbye" -wave-goodbye -disable-output 2>&1 | FileCheck %s
; RUN: opt %s %loadnewpmbye -passes="goodbye" %{bye,}-wave-goodbye -disable-output 2>&1 | FileCheck %s
;; A plugin receives all of its -plugin-arg arguments. -plugin-arg splits at
;; the first comma only, so the value may contain commas.
; RUN: opt %s %loadnewpmbye -passes="goodbye" %{bye,}-wave-goodbye %{bye,}-bye-greeting=See,you -disable-output 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-lto2 run %t.o %loadbye %loadnewpmbye -wave-goodbye -o %t -r %t.o,somefunk,plx -r %t.o,junk,plx 2>&1 | FileCheck %s
; RUN: llvm-lto2 run %t.o %loadbye %loadnewpmbye -opt-pipeline="goodbye" -wave-goodbye -o %t -r %t.o,somefunk,plx -r %t.o,junk,plx 2>&1 | FileCheck %s
; REQUIRES: plugins, examples
; UNSUPPORTED: target={{.*windows.*}}
; Plugins are currently broken on AIX, at least in the CI.
; XFAIL: target={{.*}}-aix{{.*}}
; CHECK: Bye
; BOTH:      Bye: somefunk
; BOTH-NEXT: See,you: somefunk

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}

