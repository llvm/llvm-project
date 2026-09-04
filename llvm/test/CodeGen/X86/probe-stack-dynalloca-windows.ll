; RUN: llc -O2 < %s -o /dev/null
; RUN: llc -O2 < %s | FileCheck %s
;
; Test that dynamic allocas on Windows with probe-stack attributes that are
; not callable symbol names correctly fall back to the platform stack probe.
;
; probe-stack=inline-asm is a sentinel requesting inline probing, but
; hasInlineStackProbe() returns false on Windows (Windows uses __chkstk).
; Previously, getStackProbeSymbolName() returned the sentinel as a literal
; symbol name, emitting callq "inline-asm".
;
; probe-stack="" (empty) flowed through unchecked and crashed in
; Mangler::getNameWithPrefix via X86MCInstLower::GetSymbolFromOperand.

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

; The "inline-asm" sentinel must resolve to __chkstk, not "inline-asm".
; CHECK-LABEL: test_inline_asm_sentinel:
; CHECK-NOT:   inline-asm
; CHECK:       callq __chkstk
define ptr @test_inline_asm_sentinel(i64 %n) #0 {
entry:
  %cmp = icmp ugt i64 %n, 0
  br i1 %cmp, label %alloc, label %exit
alloc:
  %buf = alloca i8, i64 %n, align 16
  ret ptr %buf
exit:
  ret ptr null
}

; An empty probe-stack value must not crash; should also resolve to __chkstk.
; CHECK-LABEL: test_empty_probe_stack:
; CHECK-NOT:   inline-asm
; CHECK:       callq __chkstk
define ptr @test_empty_probe_stack(i64 %n) #1 {
entry:
  %cmp = icmp ugt i64 %n, 0
  br i1 %cmp, label %alloc, label %exit
alloc:
  %buf = alloca i8, i64 %n, align 16
  ret ptr %buf
exit:
  ret ptr null
}

attributes #0 = { "probe-stack"="inline-asm" }
attributes #1 = { "probe-stack"="" }
