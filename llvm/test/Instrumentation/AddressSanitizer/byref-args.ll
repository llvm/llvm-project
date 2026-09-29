; RUN: opt < %s -passes=asan -S | FileCheck %s

; Test that for call instructions, the byref arguments are not
; instrumented, as no copy is implied.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.bar = type { %struct.foo }
%struct.foo = type { ptr, ptr, ptr }

; CHECK-LABEL: @func2
; CHECK-NEXT: tail call void @func1(
; CHECK-NEXT: ret void
define dso_local void @func2(ptr %foo) sanitize_address {
  tail call void @func1(ptr byref(%struct.foo) align 8 %foo) #2
  ret void
}

declare dso_local void @func1(ptr byref(%struct.foo) align 8)
