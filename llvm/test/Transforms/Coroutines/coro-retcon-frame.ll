; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

target datalayout = "p:64:64:64"

declare void @prototype_f(ptr, i1)

declare noalias ptr @allocate(i32 %size)
declare void @deallocate(ptr %ptr)
declare void @init(ptr %ptr)
declare void @use(ptr %ptr)
declare void @use_addr_val(i64 %val, ptr %addr)

define { ptr, ptr } @f(ptr %buffer) presplitcoroutine {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %proj.1 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 1
  store i64 0, ptr %proj.1, align 8
  store i64 0, ptr %proj.2, align 8
  %escape_addr = ptrtoint ptr %tmp to i64
  %id = call token @llvm.coro.id.retcon.once(i32 32, i32 8, ptr %buffer, ptr @prototype_f, ptr @allocate, ptr @deallocate)
  %hdl = call ptr @llvm.coro.begin(token %id, ptr null)
  %proj.2.2 = getelementptr inbounds { i64, i64 }, ptr %tmp, i64 0, i32 1
  call void @init(ptr %proj.1)
  call void @init(ptr %proj.2.2)
  call void @use_addr_val(i64 %escape_addr, ptr %tmp)
  %abort = call i1 (...) @llvm.coro.suspend.retcon.i1(ptr %tmp)
  br i1 %abort, label %end, label %resume

resume:
  call void @use(ptr %tmp)
  br label %end

end:
  call i1 @llvm.coro.end(ptr %hdl, i1 0)
  unreachable
}
; Make sure we don't lose writes to the frame.
; CHECK-LABEL: define { ptr, ptr } @f(ptr %buffer) {
; CHECK:  [[PROJ2:%.*]] = getelementptr inbounds { i64, i64 }, ptr %buffer, i64 0, i32 1
; CHECK:  store i64 0, ptr %buffer
; CHECK:  store i64 0, ptr [[PROJ2]]
; CHECK:  [[ESCAPED_ADDR:%.*]] = ptrtoint ptr %buffer to i64
; CHECK:  call void @init(ptr %buffer)
; CHECK:  call void @init(ptr [[PROJ2]])
; CHECK:  call void @use_addr_val(i64 [[ESCAPED_ADDR]], ptr %buffer)

; CHECK-LABEL: define internal void @f.resume.0(ptr {{.*}} %0, i1 %1) {
; CHECK: resume:
; CHECK:  call void @use(ptr %0)

declare token @llvm.coro.id.retcon.once(i32, i32, ptr, ptr, ptr, ptr)
declare ptr @llvm.coro.begin(token, ptr)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(ptr, i1)

