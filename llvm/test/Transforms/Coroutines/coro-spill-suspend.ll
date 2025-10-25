; Check that the return value of @llvm.coro.suspend gets spilled to the frame
; if it may be used across suspend points.
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s


; %sp1 should be part of the frame (the i8 value).
; CHECK: %f.Frame = type { ptr, ptr, i1, i8 }

; If the coro resumes, %sp1 is set to 0.
; CHECK-LABEL: define{{.*}} void @f.resume
; CHECK: AfterCoroSuspend:
; CHECK: %sp1.spill.addr = getelementptr inbounds %f.Frame
; CHECK: store i8 0, ptr %sp1.spill.addr

; In the coro destroy function, %sp1 is reloaded from the frame. Its value
; depends on whether the coroutine was resumed or not.
; CHECK-LABEL: define{{.*}} void @f.destroy
; CHECK: cleanup:
; CHECK: %sp1.reload.addr = getelementptr inbounds %f.Frame
; CHECK: %sp1.reload = load i8, ptr %sp1.reload.addr
; CHECK: call void @print(i8 %sp1.reload)


define ptr @f(i32 %n) presplitcoroutine {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)

  %sp1 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp1, label %suspend [i8 0, label %resume1
                                  i8 1, label %cleanup]

resume1:
  %sp2 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp2, label %suspend [i8 0, label %resume2
                                  i8 1, label %cleanup]

resume2:
  br label %cleanup

cleanup:
  ; This use of %sp1 may cross a suspend point (%sp2).
  call void @print(i8 %sp1)

  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend

suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}


declare noalias ptr @malloc(i32)
declare void @print(i8)
declare void @free(ptr)
