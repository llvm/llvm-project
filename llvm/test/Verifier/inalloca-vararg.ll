; RUN: not llvm-as %s -o /dev/null 2>&1 | FileCheck %s

declare void @h(i32, ...)
define void @i() {
  %args = alloca inalloca i32
  call void (i32, ...) @h(i32 1, i32* inalloca %args, i32 3)
; CHECK: inalloca isn't on the last argument!
  ret void
}
