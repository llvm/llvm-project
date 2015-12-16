; RUN: not llvm-as -disable-output <%s 2>&1 | FileCheck %s

define void @test() personality i32 (...)* @dummy_personality {
; CHECK: The landingpad instruction should have a consistent result type inside a function
entry:
  invoke void @dummy1()
          to label %next unwind label %unwind1

unwind1:
  %lp1 = landingpad token
            cleanup
  br label %return

next:
  invoke void @dummy2()
          to label %return unwind label %unwind2

unwind2:
  %lp2 = landingpad { i8*, i32 }
            cleanup
  br label %return

return:
  ret void
}

declare void @dummy1()
declare void @dummy2()

declare i32 @dummy_personality(...)
