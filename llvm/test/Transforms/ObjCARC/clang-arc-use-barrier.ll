; RUN: opt -passes=objc-arc -S %s | FileCheck %s

%0 = type opaque

; Make sure ARC optimizer doesn't sink @obj_retain past @llvm.objc.clang.arc.use.

; CHECK: call ptr @llvm.objc.retain
; CHECK: call void (...) @llvm.objc.clang.arc.use(
; CHECK: call ptr @llvm.objc.retain
; CHECK: call void (...) @llvm.objc.clang.arc.use(

define void @runTest() local_unnamed_addr {
  %1 = alloca ptr, align 8
  %2 = alloca ptr, align 8
  %3 = tail call ptr @foo0()
  %4 = tail call ptr @llvm.objc.retainAutoreleasedReturnValue(ptr %3)
  store ptr %3, ptr %1, align 8
  call void @foo1(ptr nonnull %1)
  %5 = load ptr, ptr %1, align 8
  %6 = call ptr @llvm.objc.retain(ptr %5)
  call void (...) @llvm.objc.clang.arc.use(ptr %3)
  call void @llvm.objc.release(ptr %3)
  store ptr %5, ptr %2, align 8
  call void @foo1(ptr nonnull %2)
  %7 = load ptr, ptr %2, align 8
  %8 = call ptr @llvm.objc.retain(ptr %7)
  call void (...) @llvm.objc.clang.arc.use(ptr %5)
  %tmp1 = load ptr, ptr %2, align 8
  call void @llvm.objc.release(ptr %5)
  call void @foo2(ptr %7)
  call void @llvm.objc.release(ptr %7)
  ret void
}

declare ptr @foo0() local_unnamed_addr
declare void @foo1(ptr) local_unnamed_addr
declare void @foo2(ptr) local_unnamed_addr

declare ptr @llvm.objc.retainAutoreleasedReturnValue(ptr) local_unnamed_addr
declare ptr @llvm.objc.retain(ptr) local_unnamed_addr
declare void @llvm.objc.clang.arc.use(...) local_unnamed_addr
declare void @llvm.objc.release(ptr) local_unnamed_addr
