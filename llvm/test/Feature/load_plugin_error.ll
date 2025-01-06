; REQUIRES: plugins, x86-registered-target
; UNSUPPORTED: target={{.*windows.*}}

; RUN: not opt < %s -load-pass-plugin=%t/nonexistent.so -disable-output 2>&1 | FileCheck %s

; RUN: opt %s -o %t.o
; RUN: not llvm-lto2 run -load-pass-plugin=%t/nonexistent.so %t.o -o %t \
; RUN:     -r %t.o,test 2>&1 | \
; RUN:   FileCheck %s

; CHECK: Could not load library {{.*}}nonexistent.so

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test() {
  ret void
}
