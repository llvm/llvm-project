  ; RUN: opt < %s -passes='module(coro-early),cgscc(coro-split),simplifycfg' -S | FileCheck %s
  
  ; When splitting a coroutine where all suspension points inside are never 
  ; suspended during runtime, i.e. await_ready() == true,
  ;
  ; In @f (entry count = 100), all coro.suspend blocks have execution count = 0.
  ; Previously, @f.resume incorrectly got copied entry count = 100.
  ; Correctly, @f.resume should receive the sum of profile execution counts across
  ; all suspension blocks, which is 0.
  
  ; CHECK-LABEL: define internal void @f.resume(
  ; CHECK-SAME:    !prof ![[RESUME_PROF:[0-9]+]]
  
  ; CHECK: ![[RESUME_PROF]] = !{!"function_entry_count", i64 0}
  
  define ptr @f(i1 %flag) presplitcoroutine !prof !0 {
  entry:
    %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
    %size = call i32 @llvm.coro.size.i32()
    %alloc = call ptr @malloc(i32 %size)
    %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
    
    ; Simulate await_ready() == true for co_await foo(1+n):
    ; Branch to continuation %cont.1 has weight 100, branch to %suspend.1 has weight 0.
    br i1 %flag, label %cont.1, label %suspend.1, !prof !1
  
  suspend.1:
    %suspend.res.1 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %suspend.res.1, label %suspend.exit [
      i8 0, label %cont.1
      i8 1, label %cleanup
    ], !prof !2
  
  cont.1:
    ; Simulate await_ready() == true for co_await foo(2+n):
    ; Branch to continuation %cont.2 has weight 100, branch to %suspend.2 has weight 0.
    br i1 %flag, label %cont.2, label %suspend.2, !prof !1
  
  suspend.2:
    %suspend.res.2 = call i8 @llvm.coro.suspend(token none, i1 false)
    switch i8 %suspend.res.2, label %suspend.exit [
      i8 0, label %cont.2
      i8 1, label %cleanup
    ], !prof !2
  
  cont.2:
    br label %cleanup
  
  cleanup:
    %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
    call void @free(ptr %mem)
    br label %suspend.exit
  
  suspend.exit:
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
  !1 = !{!"branch_weights", i32 100, i32 0}
  !2 = !{!"branch_weights", i32 0, i32 1000, i32 0}