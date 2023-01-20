; RUN: opt -opaque-pointers=0 < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

declare i8* @malloc(i64)
declare void @free(i8*)
declare void @usePointer(i8*)
declare void @usePointer2([0 x i8]*)

declare token @llvm.coro.id(i32, i8* readnone, i8* nocapture readonly, i8*)
declare i64 @llvm.coro.size.i64()
declare i8* @llvm.coro.begin(token, i8* writeonly)
declare i8 @llvm.coro.suspend(token, i1)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.free(token, i8* nocapture readonly)
declare token @llvm.coro.save(i8*)

define void @foo() presplitcoroutine {
entry:
  %a0 = alloca [0 x i8]
  %a1 = alloca i32
  %a2 = alloca [0 x i8]
  %a3 = alloca [0 x i8]
  %a4 = alloca i16
  %a5 = alloca [0 x i8]
  %a0.cast = bitcast [0 x i8]* %a0 to i8*
  %a1.cast = bitcast i32* %a1 to i8*
  %a2.cast = bitcast [0 x i8]* %a2 to i8*
  %a3.cast = bitcast [0 x i8]* %a3 to i8*
  %a4.cast = bitcast i16* %a4 to i8*
  %coro.id = call token @llvm.coro.id(i32 0, i8* null, i8* null, i8* null)
  %coro.size = call i64 @llvm.coro.size.i64()
  %coro.alloc = call i8* @malloc(i64 %coro.size)
  %coro.state = call i8* @llvm.coro.begin(token %coro.id, i8* %coro.alloc)
  %coro.save = call token @llvm.coro.save(i8* %coro.state)
  %call.suspend = call i8 @llvm.coro.suspend(token %coro.save, i1 false)
  switch i8 %call.suspend, label %suspend [
    i8 0, label %wakeup
    i8 1, label %cleanup
  ]

wakeup:                                           ; preds = %entry
  call void @usePointer(i8* %a0.cast)
  call void @usePointer(i8* %a1.cast)
  call void @usePointer(i8* %a2.cast)
  call void @usePointer(i8* %a3.cast)
  call void @usePointer(i8* %a4.cast)
  call void @usePointer2([0 x i8]* %a5)
  br label %cleanup

suspend:                                          ; preds = %cleanup, %entry
  %unused = call i1 @llvm.coro.end(i8* %coro.state, i1 false)
  ret void

cleanup:                                          ; preds = %wakeup, %entry
  %coro.memFree = call i8* @llvm.coro.free(token %coro.id, i8* %coro.state)
  call void @free(i8* %coro.memFree)
  br label %suspend
}

; CHECK:       %foo.Frame = type { void (%foo.Frame*)*, void (%foo.Frame*)*, i32, i16, i1 }

; CHECK-LABEL: @foo.resume(
; CHECK-NEXT:  entry.resume:
; CHECK-NEXT:    [[VFRAME:%.*]] = bitcast %foo.Frame* [[FRAMEPTR:%.*]] to i8*
; CHECK-NEXT:    [[TMP0:%.*]] = getelementptr inbounds [[FOO_FRAME:%.*]], %foo.Frame* [[FRAMEPTR]], i32 0, i32 0
; CHECK-NEXT:    [[A0_RELOAD_ADDR:%.*]] = bitcast void (%foo.Frame*)** [[TMP0]] to [0 x i8]*
; CHECK-NEXT:    [[A1_RELOAD_ADDR:%.*]] = getelementptr inbounds [[FOO_FRAME]], %foo.Frame* [[FRAMEPTR]], i32 0, i32 2
; CHECK-NEXT:    [[A4_RELOAD_ADDR:%.*]] = getelementptr inbounds [[FOO_FRAME]], %foo.Frame* [[FRAMEPTR]], i32 0, i32 3
; CHECK-NEXT:    [[A4_CAST5:%.*]] = bitcast i16* [[A4_RELOAD_ADDR]] to i8*
; CHECK-NEXT:    [[A3_CAST4:%.*]] = bitcast [0 x i8]* [[A0_RELOAD_ADDR]] to i8*
; CHECK-NEXT:    [[A1_CAST2:%.*]] = bitcast i32* [[A1_RELOAD_ADDR]] to i8*
; CHECK-NEXT:    call void @usePointer(i8* [[A3_CAST4]])
; CHECK-NEXT:    call void @usePointer(i8* [[A1_CAST2]])
; CHECK-NEXT:    call void @usePointer(i8* [[A3_CAST4]])
; CHECK-NEXT:    call void @usePointer(i8* [[A3_CAST4]])
; CHECK-NEXT:    call void @usePointer(i8* [[A4_CAST5]])
; CHECK-NEXT:    call void @usePointer2([0 x i8]* [[A0_RELOAD_ADDR]])
; CHECK-NEXT:    call void @free(i8* [[VFRAME]])
; CHECK-NEXT:    ret void
