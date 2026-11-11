  ; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),simplifycfg' -S | FileCheck %s
  
  ; When splitting a coroutine with PGO profile data, copying the original
  ; function entry count to the cloned .resume function underestimates its hotness
  ; when suspension points reside inside loops.
  ;
  ; In @f (entry count = 100), the coro.suspend resides inside a loop with an
  ; average trip count of 5 (branch weights 5 : 1).
  ; Previously, @f.resume just copied entry count = 100.
  ; Correctly, @f.resume calculates the sum of profile execution counts across
  ; all suspension target blocks, which is 600 because 5 suspend from the loop
  ; plue one suspend from the coroutine exit.
  
  ; CHECK-LABEL: define internal void @f.resume(
  ; CHECK-SAME:    !prof ![[RESUME_PROF:[0-9]+]]
  
  ; CHECK: ![[RESUME_PROF]] = !{!"function_entry_count", i64 600}
  
  define ptr @f() presplitcoroutine !prof !0 {
  entry:
    %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call ptr @malloc(i32 %size)
    %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
    br label %loop.body
  
  loop.body:
    %i = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
    %suspend.res = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %suspend.res, label %suspend [
      i8 0, label %loop.latch
      i8 1, label %cleanup
    ], !prof !2
  
  loop.latch:
    %i.next = add i32 %i, 1
    %cmp = icmp slt i32 %i.next, 5
    ; Loop 5 trip per entry.
    br i1 %cmp, label %loop.body, label %cleanup, !prof !1
  
  cleanup:
    %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
    call void @free(ptr %mem)
    br label %suspend
  
  suspend:
    call void @llvm.coro.end(ptr %hdl, i1 false, token none)
    ret ptr %hdl
  }
  
  declare ptr @llvm.coro.free(token, ptr)
  declare i32 @llvm.coro.size.i32()
  declare i8  @llvm.coro.suspend(token, i1)
  declare void @llvm.coro.resume(ptr)
  declare void @llvm.coro.destroy(ptr)
  declare token @llvm.coro.id(i32, ptr, ptr, ptr)
  declare ptr @llvm.coro.begin(token, ptr)
  declare void @llvm.coro.end(ptr, i1, token)
  declare noalias ptr @malloc(i32)
  declare void @free(ptr)
  
  !0 = !{!"function_entry_count", i64 100}
  !1 = !{!"branch_weights", i32 500, i32 100}
  !2 = !{!"branch_weights", i32 0, i32 1000, i32 0}
  