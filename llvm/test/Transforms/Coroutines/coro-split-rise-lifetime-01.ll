; Test lifetime-move and coro-split correctly optimize allocas that do not cross suspension points
; RUN: opt < %s -passes='cgscc(lifetime-move,coro-split),early-cse' -S | FileCheck %s

; CHECK: %large.alloca = alloca [500 x i8], align 16
; CHECK-NOT: %large.alloca.reload.addr

define void @f() presplitcoroutine {
entry:
  %large.alloca = alloca [500 x i8], align 16
  %id = call token @llvm.coro.id(i32 16, ptr null, ptr null, ptr null)
  %size = call i32 @llvm.coro.size.i32()
  %mem = call ptr @malloc(i32 %size)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr %mem)
  call void @llvm.lifetime.start.p0(ptr %large.alloca)
  %value = load i8, ptr %large.alloca, align 1
  call void @consume(i8 %value)
  %0 = call i8 @llvm.coro.suspend(token none, i1 false)
  switch i8 %0, label %exit [
    i8 0, label %suspend
    i8 1, label %cleanup
  ]

suspend:
  br label %cleanup

cleanup:
  call void @llvm.lifetime.end.p0(ptr %large.alloca)
  %1 = call ptr @llvm.coro.free(token %id, ptr %mem)
  call void @free(ptr %mem)
  br label %exit

exit:
  %2 = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  ret void
}

declare void @consume(i8)
declare ptr @malloc(i32)
declare void @free(ptr)
