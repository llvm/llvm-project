; RUN: llc < %s -O1 -mtriple=x86_64-pc-win32 | FileCheck %s

; Neither of these functions need .seh_ directives. We used to crash.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

declare i32 @__CxxFrameHandler3(...)

define void @f1() uwtable nounwind personality i32 (...)* @__CxxFrameHandler3 {
  ret void
}

; CHECK-LABEL: f1:
; CHECK-NOT: .seh_

define void @f2() uwtable {
  ret void
}

; CHECK-LABEL: f2:
; CHECK-NOT: .seh_
