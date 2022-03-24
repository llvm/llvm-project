; RUN: opt -passes=lower-global-dtors -S < %s | FileCheck %s

; Test we do not crash when reusing a pre-existing @__dso_handle global with a
; type other than i8, instead make sure we cast it.

%struct.mach_header = type { i32, i32, i32, i32, i32, i32, i32 }
@__dso_handle = external global %struct.mach_header

declare void @foo()

@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 0, void ()* @foo, i8* null }
]

; CHECK: call i32 @__cxa_atexit(void (i8*)* @call_dtors.0, i8* null, i8* bitcast (%struct.mach_header* @__dso_handle to i8*))
