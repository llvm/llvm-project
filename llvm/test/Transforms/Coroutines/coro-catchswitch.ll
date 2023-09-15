; Verifies that we can insert the spill for a PHI preceding the catchswitch
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc"

; CHECK-LABEL: define void @f(
define void @f(i1 %cond) presplitcoroutine personality i32 0 {
entry:
  %id = call token @llvm.coro.id(i32 8, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %alloc)
  br i1 %cond, label %if.else, label %if.then

if.then:
  invoke void @may_throw1()
          to label %coro.ret unwind label %catch.dispatch

if.else:
  invoke void @may_throw2()
          to label %coro.ret unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %if.else, %if.then
  %val = phi i32 [ 1, %if.then ], [ 2, %if.else ]
  %switch = catchswitch within none [label %catch] unwind label %cleanuppad

; Verifies that we split out the PHI into a separate block
; added a cleanuppad spill cleanupret unwinding into the catchswitch.

; CHECK: catch.dispatch:
; CHECK:  %val = phi i32 [ 2, %if.else ], [ 1, %if.then ]
; CHECK:  %[[Pad:.+]] = cleanuppad within none []
; CHECK:  %val.spill.addr = getelementptr inbounds %f.Frame, ptr %hdl, i32 0, i32 2
; CHECK:  store i32 %val, ptr %val.spill.addr
; CHECK:  cleanupret from %[[Pad]] unwind label %[[Switch:.+]]

; CHECK: [[Switch]]:
; CHECK: %switch = catchswitch within none [label %catch] unwind to caller

catch:                                            ; preds = %catch.dispatch
  %pad = catchpad within %switch [ptr null, i32 64, ptr null]
  catchret from %pad to label %suspend

suspend:
  %sp = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %sp, label %coro.ret [
    i8 0, label %resume
    i8 1, label %coro.ret
  ]

resume:                                   ; preds = %await2.suspend
  call void @print(i32 %val)
  br label %coro.ret

coro.ret:
  call i1 @llvm.coro.end(ptr %hdl, i1 0, token none)
    ret void

cleanuppad:
  %cpad = cleanuppad within none []
  cleanupret from %cpad unwind to caller
}

; Function Attrs: argmemonly nounwind readonly
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #1

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #2

; Function Attrs: nobuiltin
declare i32 @llvm.coro.size.i32() #4
declare ptr @llvm.coro.begin(token, ptr writeonly) #2
declare token @llvm.coro.save(ptr)
declare i8 @llvm.coro.suspend(token, i1)

declare void @may_throw1()
declare void @may_throw2()
declare void @print(i32)
declare noalias ptr @malloc(i32)
declare void @free(ptr)

declare i1 @llvm.coro.end(ptr, i1, token) #2

; Function Attrs: nobuiltin nounwind

; Function Attrs: argmemonly nounwind readonly
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #1
