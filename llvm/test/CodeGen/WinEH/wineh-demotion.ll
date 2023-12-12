; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -win-eh-prepare  < %s | FileCheck %s
; RUN: opt -mtriple=x86_64-pc-windows-msvc -S -passes=win-eh-prepare  < %s | FileCheck %s

declare i32 @__CxxFrameHandler3(...)

declare void @f()

declare i32 @g()

declare void @h(i32)

declare i1 @i()

declare void @llvm.bar() nounwind

; CHECK-LABEL: @test1(
define void @test1(i1 %B) personality ptr @__CxxFrameHandler3 {
entry:
  ; Spill slot should be inserted here
  ; CHECK: [[Slot:%[^ ]+]] = alloca
  ; Can't store for %phi at these defs because the lifetimes overlap
  ; CHECK-NOT: store
  %x = call i32 @g()
  %y = call i32 @g()
  br i1 %B, label %left, label %right
left:
  ; CHECK: left:
  ; CHECK-NEXT: store i32 %x, ptr [[Slot]]
  ; CHECK-NEXT: invoke void @f
  invoke void @f()
          to label %exit unwind label %merge
right:
  ; CHECK: right:
  ; CHECK-NEXT: store i32 %y, ptr [[Slot]]
  ; CHECK-NEXT: invoke void @f
  invoke void @f()
          to label %exit unwind label %merge
merge:
  ; CHECK: merge:
  ; CHECK-NOT: = phi
  %phi = phi i32 [ %x, %left ], [ %y, %right ]
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
  %cp = catchpad within %cs1 []
  ; CHECK: catch:
  ; CHECK: [[Reload:%[^ ]+]] = load i32, ptr [[Slot]]
  ; CHECK-NEXT: call void @h(i32 [[Reload]])
  call void @h(i32 %phi) [ "funclet"(token %cp) ]
  catchret from %cp to label %exit

exit:
  ret void
}

; CHECK-LABEL: @test2(
define void @test2(i1 %B) personality ptr @__CxxFrameHandler3 {
entry:
  br i1 %B, label %left, label %right
left:
  ; Need two stores here because %x and %y interfere so they need 2 slots
  ; CHECK: left:
  ; CHECK:   store i32 1, ptr [[Slot1:%[^ ]+]]
  ; CHECK:   store i32 1, ptr [[Slot2:%[^ ]+]]
  ; CHECK-NEXT: invoke void @f
  invoke void @f()
          to label %exit unwind label %merge.inner
right:
  ; Need two stores here because %x and %y interfere so they need 2 slots
  ; CHECK: right:
  ; CHECK-DAG:   store i32 2, ptr [[Slot1]]
  ; CHECK-DAG:   store i32 2, ptr [[Slot2]]
  ; CHECK: invoke void @f
  invoke void @f()
          to label %exit unwind label %merge.inner
merge.inner:
  ; CHECK: merge.inner:
  ; CHECK-NOT: = phi
  ; CHECK: catchswitch within none
  %x = phi i32 [ 1, %left ], [ 2, %right ]
  %cs1 = catchswitch within none [label %catch.inner] unwind label %merge.outer

catch.inner:
  %cpinner = catchpad within %cs1 []
  ; Need just one store here because only %y is affected
  ; CHECK: catch.inner:
  %z = call i32 @g() [ "funclet"(token %cpinner) ]
  ; CHECK:   store i32 %z
  ; CHECK-NEXT: invoke void @f
  invoke void @f() [ "funclet"(token %cpinner) ]
          to label %catchret.inner unwind label %merge.outer

catchret.inner:
  catchret from %cpinner to label %exit

merge.outer:
  %y = phi i32 [ %x, %merge.inner ], [ %z, %catch.inner ]
  ; CHECK: merge.outer:
  ; CHECK-NOT: = phi
  ; CHECK: catchswitch within none
  %cs2 = catchswitch within none [label %catch.outer] unwind to caller

catch.outer:
  %cpouter = catchpad within %cs2 []
  ; CHECK: catch.outer:
  ; CHECK: [[CatchPad:%[^ ]+]] = catchpad within %cs2 []
  ; Need to load x and y from two different slots since they're both live
  ; and can have different values (if we came from catch.inner)
  ; CHECK-DAG: load i32, ptr [[Slot1]]
  ; CHECK-DAG: load i32, ptr [[Slot2]]
  ; CHECK: catchret from [[CatchPad]] to label
  call void @h(i32 %x) [ "funclet"(token %cpouter) ]
  call void @h(i32 %y) [ "funclet"(token %cpouter) ]
  catchret from %cpouter to label %exit

exit:
  ret void
}

; test4: don't need stores for %phi.inner, as its only use is to feed %phi.outer
;        %phi.outer needs stores in %left, %right, and %join
; CHECK-LABEL: @test4(
define void @test4(i1 %B) personality ptr @__CxxFrameHandler3 {
entry:
  ; CHECK:      entry:
  ; CHECK:        [[Slot:%[^ ]+]] = alloca
  ; CHECK-NEXT:   br
  br i1 %B, label %left, label %right
left:
  ; CHECK: left:
  ; CHECK-NOT: store
  ; CHECK: store i32 %l, ptr [[Slot]]
  ; CHECK-NEXT: invoke void @f
  %l = call i32 @g()
  invoke void @f()
          to label %join unwind label %catchpad.inner
right:
  ; CHECK: right:
  ; CHECK-NOT: store
  ; CHECK: store i32 %r, ptr [[Slot]]
  ; CHECK-NEXT: invoke void @f
  %r = call i32 @g()
  invoke void @f()
          to label %join unwind label %catchpad.inner
catchpad.inner:
   ; CHECK: catchpad.inner:
   ; CHECK-NEXT: catchswitch within none
   %phi.inner = phi i32 [ %l, %left ], [ %r, %right ]
   %cs1 = catchswitch within none [label %catch.inner] unwind label %catchpad.outer
catch.inner:
   %cp1 = catchpad within %cs1 []
   catchret from %cp1 to label %join
join:
  ; CHECK: join:
  ; CHECK-NOT: store
  ; CHECK: store i32 %j, ptr [[Slot]]
  ; CHECK-NEXT: invoke void @f
   %j = call i32 @g()
   invoke void @f()
           to label %exit unwind label %catchpad.outer

catchpad.outer:
   ; CHECK: catchpad.outer:
   ; CHECK-NEXT: catchswitch within none
   %phi.outer = phi i32 [ %phi.inner, %catchpad.inner ], [ %j, %join ]
   %cs2 = catchswitch within none [label %catch.outer] unwind to caller
catch.outer:
   ; CHECK: catch.outer:
   ; CHECK:   [[Reload:%[^ ]+]] = load i32, ptr [[Slot]]
   ; CHECK:   call void @h(i32 [[Reload]])
   %cp2 = catchpad within %cs2 []
   call void @h(i32 %phi.outer) [ "funclet"(token %cp2) ]
   catchret from %cp2 to label %exit
exit:
   ret void
}

; CHECK-LABEL: @test5(
define void @test5() personality ptr @__CxxFrameHandler3 {
entry:
  ; need store for %phi.cleanup
  ; CHECK:      entry:
  ; CHECK:        store i32 1, ptr [[CleanupSlot:%[^ ]+]]
  ; CHECK-NEXT:   invoke void @f
  invoke void @f()
          to label %invoke.cont unwind label %cleanup

invoke.cont:
  ; need store for %phi.cleanup
  ; CHECK:      invoke.cont:
  ; CHECK-NEXT:   store i32 2, ptr [[CleanupSlot]]
  ; CHECK-NEXT:   invoke void @f
  invoke void @f()
          to label %invoke.cont2 unwind label %cleanup

cleanup:
  ; cleanup phi can be loaded at cleanup entry
  ; CHECK: cleanup:
  ; CHECK-NEXT: cleanuppad within none []
  ; CHECK: [[CleanupReload:%[^ ]+]] = load i32, ptr [[CleanupSlot]]
  %phi.cleanup = phi i32 [ 1, %entry ], [ 2, %invoke.cont ]
  %cp = cleanuppad within none []
  %b = call i1 @i() [ "funclet"(token %cp) ]
  br i1 %b, label %left, label %right

left:
  ; CHECK: left:
  ; CHECK:   call void @h(i32 [[CleanupReload]]
  call void @h(i32 %phi.cleanup) [ "funclet"(token %cp) ]
  br label %merge

right:
  ; CHECK: right:
  ; CHECK:   call void @h(i32 [[CleanupReload]]
  call void @h(i32 %phi.cleanup) [ "funclet"(token %cp) ]
  br label %merge

merge:
  ; need store for %phi.catch
  ; CHECK:      merge:
  ; CHECK-NEXT:   store i32 [[CleanupReload]], ptr [[CatchSlot:%[^ ]+]]
  ; CHECK-NEXT:   cleanupret
  cleanupret from %cp unwind label %catchswitch

invoke.cont2:
  ; need store for %phi.catch
  ; CHECK:      invoke.cont2:
  ; CHECK-NEXT:   store i32 3, ptr [[CatchSlot]]
  ; CHECK-NEXT:   invoke void @f
  invoke void @f()
          to label %exit unwind label %catchswitch

catchswitch:
  ; CHECK: catchswitch:
  ; CHECK-NEXT: catchswitch within none
  %phi.catch = phi i32 [ %phi.cleanup, %merge ], [ 3, %invoke.cont2 ]
  %cs1 = catchswitch within none [label %catch] unwind to caller

catch:
  ; CHECK: catch:
  ; CHECK:   catchpad within %cs1
  ; CHECK:   [[CatchReload:%[^ ]+]] = load i32, ptr [[CatchSlot]]
  ; CHECK:   call void @h(i32 [[CatchReload]]
  %cp2 = catchpad within %cs1 []
  call void @h(i32 %phi.catch) [ "funclet"(token %cp2) ]
  catchret from %cp2 to label %exit

exit:
  ret void
}

; We used to demote %x, but we don't need to anymore.
; CHECK-LABEL: @test6(
define void @test6() personality ptr @__CxxFrameHandler3 {
entry:
  ; CHECK: entry:
  ; CHECK: %x = invoke i32 @g()
  ; CHECK-NEXT: to label %loop unwind label %to_caller
  %x = invoke i32 @g()
          to label %loop unwind label %to_caller
to_caller:
  %cp1 = cleanuppad within none []
  cleanupret from %cp1 unwind to caller
loop:
  invoke void @f()
          to label %loop unwind label %cleanup
cleanup:
  ; CHECK: cleanup:
  ; CHECK:   call void @h(i32 %x)
  %cp2 = cleanuppad within none []
  call void @h(i32 %x) [ "funclet"(token %cp2) ]
  cleanupret from %cp2 unwind to caller
}

; CHECK-LABEL: @test7(
define void @test7() personality ptr @__CxxFrameHandler3 {
entry:
  ; %x is an EH pad phi, so gets stored in pred here
  ; CHECK: entry:
  ; CHECK:   store i32 1, ptr [[SlotX:%[^ ]+]]
  ; CHECK:   invoke void @f()
  invoke void @f()
     to label %invoke.cont unwind label %catchpad
invoke.cont:
  ; %x is an EH pad phi, so gets stored in pred here
  ; CHECK: invoke.cont:
  ; CHECK:   store i32 2, ptr [[SlotX]]
  ; CHECK:   invoke void @f()
  invoke void @f()
    to label %exit unwind label %catchpad
catchpad:
  ; %x phi should be eliminated
  ; CHECK: catchpad:
  ; CHECK-NEXT: catchswitch within none
  %x = phi i32 [ 1, %entry ], [ 2, %invoke.cont ]
  %cs1 = catchswitch within none [label %catch] unwind to caller
catch:
  ; CHECK: catch:
  ; CHECK-NEXT: %[[CatchPad:[^ ]+]] = catchpad within %cs1 []
  %cp = catchpad within %cs1 []
  %b = call i1 @i() [ "funclet"(token %cp) ]
  br i1 %b, label %left, label %right
left:
  ; Edge from %left to %join needs to be split so that
  ; the load of %x can be inserted *after* the catchret
  ; CHECK: left:
  ; CHECK-NEXT: catchret from %[[CatchPad]] to label %[[SplitLeft:[^ ]+]]
  catchret from %cp to label %join
  ; CHECK: [[SplitLeft]]:
  ; CHECK:   [[LoadX:%[^ ]+]] = load i32, ptr [[SlotX]]
  ; CHECK:   br label %join
right:
  ; Edge from %right to %join needs to be split so that
  ; the load of %y can be inserted *after* the catchret
  ; CHECK: right:
  ; CHECK:   %y = call i32 @g()
  ; CHECK:   catchret from %[[CatchPad]] to label %join
  %y = call i32 @g() [ "funclet"(token %cp) ]
  catchret from %cp to label %join
join:
  ; CHECK: join:
  ; CHECK:   %phi = phi i32 [ [[LoadX]], %[[SplitLeft]] ], [ %y, %right ]
  %phi = phi i32 [ %x, %left ], [ %y, %right ]
  call void @h(i32 %phi)
  br label %exit
exit:
  ret void
}

; CHECK-LABEL: @test8(
define void @test8() personality ptr @__CxxFrameHandler3 { entry:
  invoke void @f()
          to label %done unwind label %cleanup1
  invoke void @f()
          to label %done unwind label %cleanup2

done:
  ret void

cleanup1:
  ; CHECK: [[CleanupPad1:%[^ ]+]] = cleanuppad within none []
  ; CHECK-NEXT: call void @llvm.bar()
  ; CHECK-NEXT: cleanupret from [[CleanupPad1]]
  %cp0 = cleanuppad within none []
  br label %cleanupexit

cleanup2:
  ; CHECK: cleanuppad within none []
  ; CHECK-NEXT: call void @llvm.bar()
  ; CHECK-NEXT: unreachable
  %cp1 = cleanuppad within none []
  br label %cleanupexit

cleanupexit:
  call void @llvm.bar()
  cleanupret from %cp0 unwind label %cleanup2
}
