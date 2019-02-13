; RUN: opt -hotcoldsplit -hotcoldsplit-threshold=-1 -S < %s | FileCheck %s

; Do not outline within functions with scoped EH personalities.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.0"

; CHECK-NOT: @foo.cold.1
define void @foo() personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @"?f@@YAXXZ"() to label %normal unwind label %exception

normal:
  ret void

exception:
  %0 = cleanuppad within none []
  call void @"?terminate@@YAXXZ"() [ "funclet"(token %0) ]
  br label %continue-exception

continue-exception:
  ret void
}

declare i32 @__CxxFrameHandler3(...)

declare void @"?f@@YAXXZ"()

declare void @"?terminate@@YAXXZ"()
