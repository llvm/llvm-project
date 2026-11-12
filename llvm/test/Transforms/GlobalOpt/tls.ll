; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; RUN: opt -emulated-tls < %s -passes=globalopt -S | FileCheck %s

declare void @wait()
declare void @signal()
declare void @start_thread(ptr)

@x = internal thread_local global [100 x i32] zeroinitializer, align 16
@ip = internal global ptr null, align 8

; PR14309: GlobalOpt would think that the value of @ip is always the address of
; x[1]. However, that address is different for different threads so @ip cannot
; be replaced with a constant.

define i32 @f() {
entry:
  ; Set @ip to point to x[1] for thread 1.
  %p = call ptr @llvm.threadlocal.address(ptr @x)
  %addr = getelementptr inbounds [100 x i32], ptr %p, i64 0, i64 1
  store ptr %addr, ptr @ip, align 8

  ; Run g on a new thread.
  tail call void @start_thread(ptr @g) nounwind
  tail call void @wait() nounwind

  ; Reset x[1] for thread 1.
  store i32 0, ptr %addr, align 4

  ; Read the value of @ip, which now points at x[1] for thread 2.
  %0 = load ptr, ptr @ip, align 8

  %1 = load i32, ptr %0, align 4
  ret i32 %1

; CHECK-LABEL: @f(
; Make sure that the load from @ip hasn't been removed.
; CHECK: load ptr, ptr @ip
; CHECK: ret
}

define internal void @g() nounwind uwtable {
entry:
  ; Set @ip to point to x[1] for thread 2.
  %p = call ptr @llvm.threadlocal.address(ptr @x)
  %addr = getelementptr inbounds [100 x i32], ptr %p, i64 0, i64 1
  store ptr %addr, ptr @ip, align 8

  ; Store 50 in x[1] for thread 2.
  store i32 50, ptr %addr, align 4

  tail call void @signal() nounwind
  ret void

; CHECK-LABEL: @g(
; Make sure that the store to @ip hasn't been removed.
; CHECK: store {{.*}} @ip
; CHECK: ret
}
