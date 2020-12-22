; RUN: opt -S -simplifycfg -simplifycfg-require-and-preserve-domtree=1 < %s | FileCheck %s

declare void @Personality()
declare void @f()

; CHECK-LABEL: define void @test1()
define void @test1() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  ; CHECK: call void @f()
  invoke void @f()
    to label %exit unwind label %unreachable.unwind
exit:
  ret void
unreachable.unwind:
  cleanuppad within none []
  unreachable
}

; CHECK-LABEL: define void @test2()
define void @test2() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
    to label %exit unwind label %catch.pad
catch.pad:
  %cs1 = catchswitch within none [label %catch.body] unwind label %unreachable.unwind
  ; CHECK: catch.pad:
  ; CHECK-NEXT: catchswitch within none [label %catch.body] unwind to caller
catch.body:
  ; CHECK:      catch.body:
  ; CHECK-NEXT:   catchpad within %cs1
  ; CHECK-NEXT:   call void @f()
  ; CHECK-NEXT:   unreachable
  %catch = catchpad within %cs1 []
  call void @f()
  catchret from %catch to label %unreachable
exit:
  ret void
unreachable.unwind:
  cleanuppad within none []
  unreachable
unreachable:
  unreachable
}

; CHECK-LABEL: define void @test3()
define void @test3() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
    to label %exit unwind label %cleanup.pad
cleanup.pad:
  ; CHECK: %cleanup = cleanuppad within none []
  ; CHECK-NEXT: call void @f()
  ; CHECK-NEXT: unreachable
  %cleanup = cleanuppad within none []
  invoke void @f()
    to label %cleanup.ret unwind label %unreachable.unwind
cleanup.ret:
  ; This cleanupret should be rewritten to unreachable,
  ; and merged into the pred block.
  cleanupret from %cleanup unwind label %unreachable.unwind
exit:
  ret void
unreachable.unwind:
  cleanuppad within none []
  unreachable
}

; CHECK-LABEL: define void @test5()
define void @test5() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
          to label %exit unwind label %catch.pad

catch.pad:
  %cs1 = catchswitch within none [label %catch.body] unwind to caller

catch.body:
  %catch = catchpad within %cs1 []
  catchret from %catch to label %exit

exit:
  unreachable
}

; CHECK-LABEL: define void @test6()
define void @test6() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
          to label %exit unwind label %catch.pad

catch.pad:
  %cs1 = catchswitch within none [label %catch.body, label %catch.body] unwind to caller
  ; CHECK: catchswitch within none [label %catch.body] unwind to caller

catch.body:
  %catch = catchpad within %cs1 [i8* null, i32 0, i8* null]
  catchret from %catch to label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @test7()
define void @test7() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
          to label %exit unwind label %catch.pad

catch.pad:
  %cs1 = catchswitch within none [label %catch.body, label %catch.body2] unwind to caller
  ; CHECK: catchswitch within none [label %catch.body] unwind to caller

catch.body:
  %catch = catchpad within %cs1 [i8* null, i32 0, i8* null]
  catchret from %catch to label %exit

catch.body2:
  %catch2 = catchpad within %cs1 [i8* null, i32 0, i8* null]
  catchret from %catch2 to label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @test8()
define void @test8() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
          to label %exit unwind label %catch.pad

catch.pad:
  %cs1 = catchswitch within none [label %catch.body, label %catch.body2] unwind to caller
  ; CHECK: catchswitch within none [label %catch.body] unwind to caller

catch.body2:
  %catch2 = catchpad within %cs1 [i8* null, i32 0, i8* null]
  catchret from %catch2 to label %exit

catch.body:
  %catch = catchpad within %cs1 [i8* null, i32 0, i8* null]
  catchret from %catch to label %exit

exit:
  ret void
}

; CHECK-LABEL: define void @test9()
define void @test9() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
          to label %exit unwind label %catch.pad

catch.pad:
  %cs1 = catchswitch within none [label %catch.body, label %catch.body2] unwind to caller
  ; CHECK: catchswitch within none [label %catch.body, label %catch.body2] unwind to caller

catch.body:
  %catch = catchpad within %cs1 [i8* null, i32 0, i8* null]
  catchret from %catch to label %exit

catch.body2:
  %catch2 = catchpad within %cs1 [i8* null, i32 64, i8* null]
  catchret from %catch2 to label %exit

exit:
  ret void
}
