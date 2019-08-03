; RUN: opt -mtriple=i686-unknown-windows-msvc -S -x86-winehstate < %s | FileCheck %s

$f = comdat any

define void @f() comdat personality i32 (...)* @__CxxFrameHandler3 {
  invoke void @g() to label %return unwind label %unwind
return:
  ret void
unwind:
  %pad = cleanuppad within none []
  cleanupret from %pad unwind to caller
}

declare void @g()
declare i32 @__CxxFrameHandler3(...)

; CHECK: define internal i32 @"__ehhandler$f"(i8* %0, i8* %1, i8* %2, i8* %3){{ .+}} comdat($f) {
