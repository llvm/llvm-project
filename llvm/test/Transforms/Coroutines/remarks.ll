; Test coroutine remarks.
; RUN: opt < %s --disable-output -S -passes='default<O1>' \
; RUN: --pass-remarks="coro-split|coro-elide" \
; RUN: --pass-remarks-missed="coro-split|coro-elide" \
; RUN: --pass-remarks-with-hotness 2>&1 | FileCheck %s

; CHECK: Split 'foo' (frame_size=24, align=8) (hotness: 400)
; CHECK: 'foo' not elided in 'bar' (frame_size=24, align=8) (hotness: 100)
; CHECK: 'foo' elided in 'baz' (frame_size=24, align=8) (hotness: 200)

define ptr @foo() presplitcoroutine !prof !0 {
entry:
  %id = call token @llvm.coro.id(i32 0, ptr null, ptr null, ptr null)
  %need.dyn.alloc = call i1 @llvm.coro.alloc(token %id)
  br i1 %need.dyn.alloc, label %dyn.alloc, label %coro.begin
dyn.alloc:
  %size = call i32 @llvm.coro.size.i32()
  %alloc = call ptr @malloc(i32 %size)
  br label %coro.begin
coro.begin:
  %phi = phi ptr [ null, %entry ], [ %alloc, %dyn.alloc ]
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %phi)
  call void @print(i32 0)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %suspend [i8 0, label %resume
                                i8 1, label %cleanup]
resume:
  call void @print(i32 1)
  br label %cleanup

cleanup:
  %mem = call ptr @llvm.coro.free(token %id, ptr %hdl)
  call void @free(ptr %mem)
  br label %suspend
suspend:
  call void @llvm.coro.end(ptr %hdl, i1 0, token none)
  ret ptr %hdl
}

define i32 @bar() !prof !1 {
entry:
  %hdl = call ptr @foo()
  call void @llvm.coro.resume(ptr %hdl)
  ret i32 0
}

define i32 @baz() !prof !2 {
entry:
  %hdl = call ptr @foo()
  call void @llvm.coro.destroy(ptr %hdl)
  ret i32 0
}

declare ptr @llvm.coro.free(token, ptr)
declare i32 @llvm.coro.size.i32()
declare i8  @llvm.coro.suspend(token, i1)
declare void @llvm.coro.resume(ptr)
declare void @llvm.coro.destroy(ptr)

declare token @llvm.coro.id(i32, ptr, ptr, ptr)
declare i1 @llvm.coro.alloc(token)
declare ptr @llvm.coro.begin(token, ptr)
declare void @llvm.coro.end(ptr, i1, token)

declare noalias ptr @malloc(i32)
declare void @print(i32)
declare void @free(ptr)

!0 = !{!"function_entry_count", i64 400}
!1 = !{!"function_entry_count", i64 100}
!2 = !{!"function_entry_count", i64 200}
