; RUN: llc -mcpu=core < %s | FileCheck %s

target triple = "i686-apple-darwin"

declare tailcc void @foo(i32, i32, i32, i32, i32, i32)
declare ptr @bar(ptr)

define tailcc void @hoge(i32 %b) nounwind {
; Do not overwrite pushed callee-save registers
; CHECK: pushl
; CHECK: subl $[[SIZE:[0-9]+]], %esp
; CHECK-NOT: [[SIZE]](%esp)
  %a = alloca i32
  store i32 0, ptr %a
  %d = tail call ptr @bar(ptr %a) nounwind
  store i32 %b, ptr %d
  tail call tailcc void @foo(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6) nounwind
  ret void
}
