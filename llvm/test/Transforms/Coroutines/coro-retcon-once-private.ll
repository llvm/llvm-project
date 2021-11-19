; RUN: opt < %s -enable-coroutines -passes='default<O0>' -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; CHECK: define internal { i8*, i32 } @f(i8* %buffer, i32* %array)
; CHECK-NEXT: entry:
; CHECK-NEXT:  unreachable

define internal {i8*, i32} @f(i8* %buffer, i32* %array) {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, i8* %buffer, i8* bitcast (void (i8*, i1)* @prototype to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %load = load i32, i32* %array
  %load.pos = icmp sgt i32 %load, 0
  br i1 %load.pos, label %pos, label %neg

pos:
  %unwind0 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %load)
  br i1 %unwind0, label %cleanup, label %pos.cont

pos.cont:
  store i32 0, i32* %array, align 4
  br label %cleanup

neg:
  %unwind1 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 0)
  br i1 %unwind1, label %cleanup, label %neg.cont

neg.cont:
  store i32 10, i32* %array, align 4
  br label %cleanup

cleanup:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}

declare token @llvm.coro.id.retcon.once(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(i8*, i1)
declare i8* @llvm.coro.prepare.retcon(i8*)

declare void @prototype(i8*, i1 zeroext)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)
