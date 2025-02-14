; Tests that coro-split pass splits the coroutine into f, f.resume and f.destroy
; RUN: opt < %s -passes='cgscc(coro-split),simplifycfg,early-cse' -S | FileCheck %s

define ptr @test_simple(ptr noalias dereferenceable(32) %0) presplitcoroutine {
entry:
  %call.aggresult = alloca <{ i64, i64, i64, i64, i64 }>, align 8
  %1 = alloca <{ i64, i64, i64, i64, i64 }>, align 8
  %2 = call token (i32, i32, ptr, ptr, ptr, ptr, ...) @llvm.coro.id.retcon.once(i32 32, i32 8, ptr %0, ptr @prototype, ptr @swift_coroFrameAlloc, ptr @free, i64 123)
  %3 = call ptr @llvm.coro.begin(token %2, ptr null)
  call swiftcc void @marker(i32 1000)
  call void @llvm.lifetime.start.p0(i64 40, ptr %call.aggresult)
  call swiftcc void @val(ptr noalias nocapture sret(<{ i64, i64, i64, i64, i64 }>) %call.aggresult)
  %call.aggresult.elt = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %call.aggresult, i32 0, i32 0
  %4 = load i64, ptr %call.aggresult.elt, align 8
  %call.aggresult.elt1 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %call.aggresult, i32 0, i32 1
  %5 = load i64, ptr %call.aggresult.elt1, align 8
  %call.aggresult.elt2 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %call.aggresult, i32 0, i32 2
  %6 = load i64, ptr %call.aggresult.elt2, align 8
  %call.aggresult.elt3 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %call.aggresult, i32 0, i32 3
  %7 = load i64, ptr %call.aggresult.elt3, align 8
  %call.aggresult.elt4 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %call.aggresult, i32 0, i32 4
  %8 = load i64, ptr %call.aggresult.elt4, align 8
  call void @llvm.lifetime.end.p0(i64 40, ptr %call.aggresult)
  %9 = call i1 (...) @llvm.coro.suspend.retcon.i1()
  br i1 %9, label %11, label %10

10:                                               ; preds = %entry
  call swiftcc void @marker(i32 2000)
  %.elt = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %1, i32 0, i32 0
  store i64 %4, ptr %.elt, align 8
  %.elt5 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %1, i32 0, i32 1
  store i64 %5, ptr %.elt5, align 8
  %.elt6 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %1, i32 0, i32 2
  store i64 %6, ptr %.elt6, align 8
  %.elt7 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %1, i32 0, i32 3
  store i64 %7, ptr %.elt7, align 8
  %.elt8 = getelementptr inbounds <{ i64, i64, i64, i64, i64 }>, ptr %1, i32 0, i32 4
  store i64 %8, ptr %.elt8, align 8
  call swiftcc void @use(ptr noalias nocapture dereferenceable(40) %1)
  br label %coro.end

11:                                               ; preds = %entry
  call swiftcc void @marker(i32 3000)
  br label %coro.end

coro.end:                                         ; preds = %10, %11
  %12 = call i1 @llvm.coro.end(ptr %3, i1 false, token none)
  unreachable
}

declare void @free(ptr)
declare ptr @swift_coroFrameAlloc(i64, i64)
declare swiftcc void @marker(i32) #1
declare swiftcc void @use(ptr noalias nocapture dereferenceable(40))
declare swiftcc void @val(ptr noalias nocapture sret(<{ i64, i64, i64, i64, i64 }>))
declare void @prototype(ptr, i1 zeroext)

; CHECK-LABEL: @test_simple(
; CHECK-NEXT:  entry:
; CHECK:    call ptr @swift_coroFrameAlloc(i64 40, i64 123)