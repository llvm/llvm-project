; RUN: opt < %s -passes=asan -S | FileCheck %s
; Test that for call instructions, the by-value arguments are instrumented.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.bar = type { %struct.foo }
%struct.foo = type { i8*, i8*, i8* }
define dso_local void @func2(%struct.foo* %foo) sanitize_address {
; CHECK-LABEL: @func2
  tail call void @func1(%struct.foo* byval(%struct.foo) align 8 %foo) #2
; CHECK: call void @__asan_report_load
  ret void
; CHECK: ret void
}
declare dso_local void @func1(%struct.foo* byval(%struct.foo) align 8)

!0 = !{i32 1, !"wchar_size", i32 4}
