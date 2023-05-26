; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

declare ptr @malloc(i64)
declare void @free(ptr)
declare void @usePointer(ptr)
declare void @usePointer2(ptr)

declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr)
declare i64 @llvm.coro.size.i64()
declare ptr @llvm.coro.begin(token, ptr writeonly)
declare i8 @llvm.coro.suspend(token, i1)
declare i1 @llvm.coro.end(ptr, i1)
declare ptr @llvm.coro.free(token, ptr nocapture readonly)
declare token @llvm.coro.save(ptr)

define void @foo() presplitcoroutine {
entry:
  %a0 = alloca [0 x i8]
  %a1 = alloca i32
  %a2 = alloca [0 x i8]
  %a3 = alloca [0 x i8]
  %a4 = alloca i16
  %a5 = alloca [0 x i8]
  %coro.id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %coro.size = call i64 @llvm.coro.size.i64()
  %coro.alloc = call ptr @malloc(i64 %coro.size)
  %coro.state = call ptr @llvm.coro.begin(token %coro.id, ptr %coro.alloc)
  %coro.save = call token @llvm.coro.save(ptr %coro.state)
  %call.suspend = call i8 @llvm.coro.suspend(token %coro.save, i1 false)
  switch i8 %call.suspend, label %suspend [
    i8 0, label %wakeup
    i8 1, label %cleanup
  ]

wakeup:                                           ; preds = %entry
  call void @usePointer(ptr %a0)
  call void @usePointer(ptr %a1)
  call void @usePointer(ptr %a2)
  call void @usePointer(ptr %a3)
  call void @usePointer(ptr %a4)
  call void @usePointer2(ptr %a5)
  br label %cleanup

suspend:                                          ; preds = %cleanup, %entry
  %unused = call i1 @llvm.coro.end(ptr %coro.state, i1 false)
  ret void

cleanup:                                          ; preds = %wakeup, %entry
  %coro.memFree = call ptr @llvm.coro.free(token %coro.id, ptr %coro.state)
  call void @free(ptr %coro.memFree)
  br label %suspend
}

; CHECK:       %foo.Frame = type { ptr, ptr, i32, i16, i1 }

; CHECK-LABEL: @foo.resume(
; CHECK-NEXT:  entry.resume:
; CHECK-NEXT:    [[A1_RELOAD_ADDR:%.*]] = getelementptr inbounds [[FOO_FRAME:%foo.Frame]], ptr [[FRAMEPTR:%.*]], i32 0, i32 2
; CHECK-NEXT:    [[A4_RELOAD_ADDR:%.*]] = getelementptr inbounds [[FOO_FRAME]], ptr [[FRAMEPTR]], i32 0, i32 3
; CHECK-NEXT:    call void @usePointer(ptr [[FRAMEPTR]])
; CHECK-NEXT:    call void @usePointer(ptr [[A1_RELOAD_ADDR]])
; CHECK-NEXT:    call void @usePointer(ptr [[FRAMEPTR]])
; CHECK-NEXT:    call void @usePointer(ptr [[FRAMEPTR]])
; CHECK-NEXT:    call void @usePointer(ptr [[A4_RELOAD_ADDR]])
; CHECK-NEXT:    call void @usePointer2(ptr [[FRAMEPTR]])
; CHECK-NEXT:    call void @free(ptr [[FRAMEPTR]])
; CHECK-NEXT:    ret void
