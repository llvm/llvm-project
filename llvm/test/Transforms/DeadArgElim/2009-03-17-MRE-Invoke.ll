; RUN: opt < %s -passes=deadargelim | llvm-dis
; PR3807

define internal { i32, i32 } @foo() {
  ret {i32,i32} {i32 42, i32 4}
}

define i32 @bar() personality i32 (...)* @__gxx_personality_v0 {
  %x = invoke {i32,i32} @foo() to label %T unwind label %T2
T:
  %y = extractvalue {i32,i32} %x, 1
  ret i32 %y
T2:
  %exn = landingpad {i8*, i32}
            cleanup
  unreachable
}

define i32 @bar2() personality i32 (...)* @__gxx_personality_v0 {
entry:
  %x = invoke {i32,i32} @foo() to label %T unwind label %T2
T:
  %PN = phi i32 [0, %entry]
  %y = extractvalue {i32,i32} %x, 1
  ret i32 %y
T2:
  %exn = landingpad {i8*, i32}
            cleanup
  unreachable
}

declare i32 @__gxx_personality_v0(...)
