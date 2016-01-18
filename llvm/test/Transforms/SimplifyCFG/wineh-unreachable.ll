; RUN: opt -S -simplifycfg < %s | FileCheck %s

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
  cleanuppad []
  unreachable  
}

; CHECK-LABEL: define void @test2()
define void @test2() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
    to label %exit unwind label %catch.pad
catch.pad:
  ; CHECK: catchpad []
  ; CHECK-NEXT: to label %catch.body unwind label %catch.end
  %catch = catchpad []
    to label %catch.body unwind label %catch.end
catch.body:
  ; CHECK:      catch.body:
  ; CHECK-NEXT:   call void @f()
  ; CHECK-NEXT:   unreachable
  call void @f()
  catchret %catch to label %unreachable
catch.end:
  ; CHECK: catch.end:
  ; CHECK-NEXT: catchendpad unwind to caller
  catchendpad unwind label %unreachable.unwind
exit:
  ret void
unreachable.unwind:
  cleanuppad []
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
  ; CHECK: %cleanup = cleanuppad []
  ; CHECK-NEXT: call void @f()
  ; CHECK-NEXT: unreachable
  %cleanup = cleanuppad []
  invoke void @f()
    to label %cleanup.ret unwind label %cleanup.end
cleanup.ret:
  ; This cleanupret should be rewritten to unreachable,
  ; and merged into the pred block.
  cleanupret %cleanup unwind label %unreachable.unwind
cleanup.end:
  ; This cleanupendpad should be rewritten to unreachable,
  ; causing the invoke to be rewritten to a call.
  cleanupendpad %cleanup unwind label %unreachable.unwind
exit:
  ret void
unreachable.unwind:
  cleanuppad []
  unreachable
}

; CHECK-LABEL: define void @test4()
define void @test4() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
    to label %exit unwind label %terminate.pad
terminate.pad:
  ; CHECK: terminatepad [] unwind to caller
  terminatepad [] unwind label %unreachable.unwind
exit:
  ret void
unreachable.unwind:
  cleanuppad []
  unreachable
}

; CHECK-LABEL: define void @test5()
define void @test5() personality i8* bitcast (void ()* @Personality to i8*) {
entry:
  invoke void @f()
          to label %exit unwind label %catch.pad

catch.pad:
  %catch = catchpad []
          to label %catch.body unwind label %catch.end

catch.body:
  catchret %catch to label %exit

catch.end:
  catchendpad unwind to caller

exit:
  unreachable
}
