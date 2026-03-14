; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff -w %t.orig %t.echo
; RUN: llvm-as < %s | llvm-c-test --test-diagnostic-handler 2>&1 | FileCheck %s
; CHECK: Diagnostic handler was not called while loading module
