; RUN: not llvm-as %s -disable-output 2>&1 | FileCheck %s


declare void @doit(ptr inalloca(i64) %a)

define void @a() {
entry:
  %a = alloca [2 x i32]
  call void @doit(ptr inalloca(i64) %a)
; CHECK: inalloca argument for call has mismatched alloca
  ret void
}
