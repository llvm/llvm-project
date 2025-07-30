; RUN: not llc -mtriple=x86_64-unknown-linux-gnu < %s 2>&1 | FileCheck %s

; CHECK: LLVM ERROR: cannot lower invokes with arbitrary operand bundles: foo

declare void @g()
declare i32 @__gxx_personality_v0(...)

define void @f(i32 %arg) personality ptr @__gxx_personality_v0 {
  invoke void @g() [ "foo"(i32 %arg) ]
    to label %cont unwind label %lpad

lpad:
  %l = landingpad {ptr, i32}
    cleanup
  resume {ptr, i32} %l

cont:
  ret void
}
