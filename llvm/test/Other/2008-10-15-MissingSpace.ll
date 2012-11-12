; RUN: llvm-as < %s | llvm-dis | not grep "void@"
; PR2894
declare void @g()
define void @f() {
  invoke void @g() to label %c unwind label %c
c:
  %exn = landingpad {i8*, i32} personality i32 (...)* @__gxx_personality_v0
            cleanup
  ret void
}

declare i32 @__gxx_personality_v0(...)
