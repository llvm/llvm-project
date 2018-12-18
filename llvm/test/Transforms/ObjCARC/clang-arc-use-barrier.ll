; RUN: opt -objc-arc -S %s | FileCheck %s

%0 = type opaque

; Make sure ARC optimizer doesn't sink @obj_retain past @llvm.objc.clang.arc.use.

; CHECK: call i8* @llvm.objc.retain
; CHECK: call void (...) @llvm.objc.clang.arc.use(
; CHECK: call i8* @llvm.objc.retain
; CHECK: call void (...) @llvm.objc.clang.arc.use(

define void @runTest() local_unnamed_addr {
  %1 = alloca %0*, align 8
  %2 = alloca %0*, align 8
  %3 = tail call %0* @foo0()
  %4 = bitcast %0* %3 to i8*
  %5 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %4)
  store %0* %3, %0** %1, align 8
  call void @foo1(%0** nonnull %1)
  %6 = load %0*, %0** %1, align 8
  %7 = bitcast %0* %6 to i8*
  %8 = call i8* @llvm.objc.retain(i8* %7)
  call void (...) @llvm.objc.clang.arc.use(%0* %3)
  call void @llvm.objc.release(i8* %4)
  store %0* %6, %0** %2, align 8
  call void @foo1(%0** nonnull %2)
  %9 = load %0*, %0** %2, align 8
  %10 = bitcast %0* %9 to i8*
  %11 = call i8* @llvm.objc.retain(i8* %10)
  call void (...) @llvm.objc.clang.arc.use(%0* %6)
  %tmp1 = load %0*, %0** %2, align 8
  call void @llvm.objc.release(i8* %7)
  call void @foo2(%0* %9)
  call void @llvm.objc.release(i8* %10)
  ret void
}

declare %0* @foo0() local_unnamed_addr
declare void @foo1(%0**) local_unnamed_addr
declare void @foo2(%0*) local_unnamed_addr

declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*) local_unnamed_addr
declare i8* @llvm.objc.retain(i8*) local_unnamed_addr
declare void @llvm.objc.clang.arc.use(...) local_unnamed_addr
declare void @llvm.objc.release(i8*) local_unnamed_addr
