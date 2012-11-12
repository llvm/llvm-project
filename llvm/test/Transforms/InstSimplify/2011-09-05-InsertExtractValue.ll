; RUN: opt < %s -instsimplify -S | FileCheck %s

declare void @bar()

define void @test1() {
entry:
  invoke void @bar() to label %cont unwind label %lpad
cont:
  ret void
lpad:
  %ex = landingpad { i8*, i32 } personality i32 (i32, i64, i8*, i8*)* @__gxx_personality_v0 cleanup
  %exc_ptr = extractvalue { i8*, i32 } %ex, 0
  %filter = extractvalue { i8*, i32 } %ex, 1
  %exc_ptr2 = insertvalue { i8*, i32 } undef, i8* %exc_ptr, 0
  %filter2 = insertvalue { i8*, i32 } %exc_ptr2, i32 %filter, 1
  resume { i8*, i32 } %filter2
; CHECK: @test1
; CHECK-NOT: extractvalue
; CHECK-NOT: insertvalue
}

declare i32 @__gxx_personality_v0(i32, i64, i8*, i8*)

define { i8, i32 } @test2({ i8*, i32 } %x) {
  %ex = extractvalue { i8*, i32 } %x, 1
  %ins = insertvalue { i8, i32 } undef, i32 %ex, 1
  ret { i8, i32 } %ins
; CHECK: @test2
}
