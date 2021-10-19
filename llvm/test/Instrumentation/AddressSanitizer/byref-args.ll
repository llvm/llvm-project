; RUN: opt < %s -passes='asan-function-pipeline' -S | FileCheck %s

; Test that for call instructions, the byref arguments are not
; instrumented, as no copy is implied.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.bar = type { %struct.foo }
%struct.foo = type { i8*, i8*, i8* }

; CHECK-LABEL: @func2
; CHECK-NEXT: tail call void @func1(
; CHECK-NEXT: ret void
define dso_local void @func2(%struct.foo* %foo) sanitize_address {
  tail call void @func1(%struct.foo* byref(%struct.foo) align 8 %foo) #2
  ret void
}

declare dso_local void @func1(%struct.foo* byref(%struct.foo) align 8)
