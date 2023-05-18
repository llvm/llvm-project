; RUN: opt -mtriple=i686-unknown-windows-msvc -S -x86-winehstate < %s | FileCheck %s

$f = comdat any

define void @f() comdat personality ptr @__CxxFrameHandler3 {
  invoke void @g() to label %return unwind label %unwind
return:
  ret void
unwind:
  %pad = cleanuppad within none []
  cleanupret from %pad unwind to caller
}

declare void @g()
declare i32 @__CxxFrameHandler3(...)

; CHECK: define internal i32 @"__ehhandler$f"(ptr %0, ptr %1, ptr %2, ptr %3){{ .+}} comdat($f) {
