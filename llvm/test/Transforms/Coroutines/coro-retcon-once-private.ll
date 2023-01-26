; RUN: opt < %s -passes='default<O0>' -S | FileCheck %s

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; CHECK-NOT: define

define internal {ptr, i32} @f(ptr %buffer, ptr %array) {
entry:
  %id = call token @llvm.coro.id.retcon.once(i32 8, i32 8, ptr %buffer, ptr @prototype, ptr @allocate, ptr @deallocate)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %load = load i32, ptr %array
  %load.pos = icmp sgt i32 %load, 0
  br i1 %load.pos, label %pos, label %neg

pos:
  %unwind0 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 %load)
  br i1 %unwind0, label %cleanup, label %pos.cont

pos.cont:
  store i32 0, ptr %array, align 4
  br label %cleanup

neg:
  %unwind1 = call i1 (...) @llvm.coro.suspend.retcon.i1(i32 0)
  br i1 %unwind1, label %cleanup, label %neg.cont

neg.cont:
  store i32 10, ptr %array, align 4
  br label %cleanup

cleanup:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  unreachable
}

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(ptr, i1)
declare ptr @llvm.coro.prepare.retcon(ptr)

declare void @prototype(ptr, i1 zeroext)

declare noalias ptr @allocate(i32 %size)
declare void @deallocate(ptr %ptr)
