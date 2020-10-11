; RUN: opt < %s -coro-split -S | FileCheck %s
; RUN: opt < %s -passes=coro-split -S | FileCheck %s

target datalayout = "p:64:64:64"

declare void @prototype_f(i8*, i1)

declare noalias i8* @allocate(i32 %size)
declare void @deallocate(i8* %ptr)
declare void @init(i64 *%ptr)
declare void @use(i8* %ptr)
declare void @use_addr_val(i64 %val, {i64, i64}*%addr)

define { i8*, {i64, i64}* } @f(i8* %buffer) "coroutine.presplit"="1" {
entry:
  %tmp = alloca { i64, i64 }, align 8
  %proj.1 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 0
  %proj.2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 1
  store i64 0, i64* %proj.1, align 8
  store i64 0, i64* %proj.2, align 8
  %cast = bitcast { i64, i64 }* %tmp to i8*
  %escape_addr = ptrtoint {i64, i64}* %tmp to i64
  %id = call token @llvm.coro.id.retcon.once(i32 32, i32 8, i8* %buffer, i8* bitcast (void (i8*, i1)* @prototype_f to i8*), i8* bitcast (i8* (i32)* @allocate to i8*), i8* bitcast (void (i8*)* @deallocate to i8*))
  %hdl = call i8* @llvm.coro.begin(token %id, i8* null)
  %proj.2.2 = getelementptr inbounds { i64, i64 }, { i64, i64 }* %tmp, i64 0, i32 1
  call void @init(i64 * %proj.1)
  call void @init(i64 * %proj.2.2)
  call void @use_addr_val(i64 %escape_addr, {i64, i64}* %tmp)
  %abort = call i1 (...) @llvm.coro.suspend.retcon.i1({i64, i64}* %tmp)
  br i1 %abort, label %end, label %resume

resume:
  call void @use(i8* %cast)
  br label %end

end:
  call i1 @llvm.coro.end(i8* %hdl, i1 0)
  unreachable
}
; Make sure we don't lose writes to the frame.
; CHECK-LABEL: define { i8*, { i64, i64 }* } @f(i8* %buffer) {
; CHECK:  [[FRAMEPTR:%.*]] = bitcast i8* %buffer to %f.Frame*
; CHECK:  [[TMP:%.*]] = getelementptr inbounds %f.Frame, %f.Frame* [[FRAMEPTR]], i32 0, i32 0
; CHECK:  [[PROJ1:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[TMP]], i64 0, i32 0
; CHECK:  [[PROJ2:%.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* [[TMP]], i64 0, i32 1
; CHECK:  store i64 0, i64* [[PROJ1]]
; CHECK:  store i64 0, i64* [[PROJ2]]
; CHECK:  [[ESCAPED_ADDR:%.*]] = ptrtoint { i64, i64 }* [[TMP]] to i64
; CHECK:  call void @init(i64* [[PROJ1]])
; CHECK:  call void @init(i64* [[PROJ2]])
; CHECK:  call void @use_addr_val(i64 [[ESCAPED_ADDR]], { i64, i64 }* [[TMP]])

; CHECK-LABEL: define internal void @f.resume.0(i8* {{.*}} %0, i1 %1) {
; CHECK:  [[FRAMEPTR:%.*]] = bitcast i8* %0 to %f.Frame*
; CHECK:  [[TMP:%.*]] = getelementptr inbounds %f.Frame, %f.Frame* [[FRAMEPTR]], i32 0, i32 0
; CHECK: resume:
; CHECK:  [[CAST:%.*]] = bitcast { i64, i64 }* [[TMP]] to i8*
; CHECK:  call void @use(i8* [[CAST]])

declare token @llvm.coro.id.retcon.once(i32, i32, i8*, i8*, i8*, i8*)
declare i8* @llvm.coro.begin(token, i8*)
declare i1 @llvm.coro.suspend.retcon.i1(...)
declare i1 @llvm.coro.end(i8*, i1)

