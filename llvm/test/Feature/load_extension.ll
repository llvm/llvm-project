; REQUIRES: x86-registered-target
; RUN: opt %s %loadnewpmbye %loadbye -passes="goodbye" -wave-goodbye -disable-output 2>&1 | FileCheck %s
;; A plugin receives all of its -plugin-arg arguments. -plugin-arg splits at
;; the first comma only, so the value may contain commas.
; RUN: opt %s %loadnewpmbye -passes="goodbye" -plugin-arg=Bye,-wave-goodbye -plugin-arg=Bye,-bye-greeting=See,you -disable-output 2>&1 | FileCheck %s --check-prefix=BOTH
; RUN: not opt %s %loadnewpmbye -plugin-arg=Bye -disable-output 2>&1 | FileCheck %s --check-prefix=NOCOMMA
; RUN: not opt %s %loadnewpmbye -plugin-arg=Nope,-x -disable-output 2>&1 | FileCheck %s --check-prefix=UNKNOWN
;; Two plugins with the same name are an error.
; RUN: %if !linked-bye %{ not opt %s %loadnewpmbye %loadnewpmbye -disable-output 2>&1 | FileCheck %s --check-prefix=DUP %}
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
; NOCOMMA: expected <plugin>,<arg> in -plugin-arg=Bye{{$}}
; UNKNOWN: no pass plugin named 'Nope' is loaded, in -plugin-arg=Nope,-x
; DUP: multiple pass plugins are named 'Bye'{{$}}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define ptr @somefunk() {
  ret ptr @junk
}

