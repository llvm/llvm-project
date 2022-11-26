; RUN: opt -passes=lower-global-dtors -S < %s | FileCheck %s

; Test we do not crash when reusing a pre-existing @__dso_handle global with a
; type other than i8, instead make sure we cast it.

%struct.mach_header = type { i32, i32, i32, i32, i32, i32, i32 }
@__dso_handle = external global %struct.mach_header

declare void @foo()

@llvm.global_dtors = appending global [1 x { i32, ptr, ptr }] [
  { i32, ptr, ptr } { i32 0, ptr @foo, ptr null }
]

; CHECK: call i32 @__cxa_atexit(ptr @call_dtors.0, ptr null, ptr @__dso_handle)
