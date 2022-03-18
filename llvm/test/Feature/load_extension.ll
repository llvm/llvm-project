; REQUIRES: x86-registered-target
; RUN: opt %s %loadbye -goodbye -wave-goodbye -disable-output -enable-new-pm=0 2>&1 | FileCheck %s
; RUN: opt %s %loadnewpmbye %loadbye -passes="goodbye" -wave-goodbye -disable-output 2>&1 | FileCheck %s
; RUN: opt %s %loadnewpmbye -passes="goodbye" -wave-goodbye -disable-output 2>&1 | FileCheck %s
; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-lto2 run %t.o %loadbye -wave-goodbye -use-new-pm=0 -o %t -r %t.o,somefunk,plx -r %t.o,junk,plx 2>&1 | FileCheck %s
; RUN: llvm-lto2 run %t.o %loadbye %loadnewpmbye -wave-goodbye -use-new-pm -o %t -r %t.o,somefunk,plx -r %t.o,junk,plx 2>&1 | FileCheck %s
; RUN: llvm-lto2 run %t.o %loadbye %loadnewpmbye -opt-pipeline="goodbye" -wave-goodbye -use-new-pm -o %t -r %t.o,somefunk,plx -r %t.o,junk,plx 2>&1 | FileCheck %s
; REQUIRES: plugins, examples
; UNSUPPORTED: windows
; CHECK: Bye
;
; Specifying a new PM pass plugin with the old PM is an error.
; RUN: ! opt %s %loadnewpmbye -goodbye -wave-goodbye -disable-output -enable-new-pm=0 2>&1 | FileCheck %s --check-prefix=ERROR
; ERROR: load-pass-plugin specified with legacy PM.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
@junk = global i32 0

define i32* @somefunk() {
  ret i32* @junk
}

