; RUN: opt %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

; Verify that memory attributes are not copied blindly to generated functions.

declare ptr @aligned_alloc(i64, i64)
declare void @free(ptr %mem)

define ptr @f(i32 %a, i32 %b) presplitcoroutine memory(none) "keep" {
entry:
  %promise = alloca i32
  %id = call token @llvm.coro.id(i32 0, ptr %promise, ptr null, ptr null)
  %size = call i64 @llvm.coro.size.i64()
  %align = call i64 @llvm.coro.align.i64()
  %alloc = call ptr @aligned_alloc(i64 %size, i64 %align)
  %hdl = call noalias ptr @llvm.coro.begin(token %id, ptr %alloc)
  %s0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %s0, label %suspend [i8 0, label %one_block
                                 i8 1, label %cleanup]
one_block:
  %c = add nuw i32 %a, %b
  store i32 %c, ptr %promise
  %s1 = call i8 @llvm.coro.suspend(token none, i1 true)
  switch i8 %s1, label %suspend [i8 0, label %trap
                                 i8 1, label %cleanup]
cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  %unused = call i1 @llvm.coro.end(ptr %hdl, i1 false, token none)
  ret ptr %hdl
trap:
  call void @llvm.trap()
  unreachable
}

; CHECK: @f.resume({{.*}}) #[[RESUME_ATTRS:[[:alnum:]]+]]
; CHECK: @f.destroy({{.*}}) #[[RESUME_ATTRS]]
; CHECK: @f.cleanup({{.*}}) #[[RESUME_ATTRS]]
; CHECK: attributes #[[RESUME_ATTRS]] = { "keep" }
