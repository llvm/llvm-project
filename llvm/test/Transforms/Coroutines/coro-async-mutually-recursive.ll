; RUN: opt < %s -passes='default<O2>' -S | FileCheck --check-prefixes=CHECK %s
; RUN: opt < %s -O0 -S | FileCheck --check-prefixes=CHECK-O0 %s


; CHECK-NOT: llvm.coro.suspend.async
; CHECK-O0-NOT: llvm.coro.suspend.async

; This test used to crash during updating the call graph in coro splitting.

target datalayout = "p:64:64:64"

%swift.async_func_pointer = type <{ i32, i32 }>

@"$s1d3fooyySbYaFTu" = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1d3fooyySbYaF" to i64), i64 ptrtoint (ptr @"$s1d3fooyySbYaFTu" to i64)) to i32), i32 16 }>
@"$s1d3baryySbYaFTu" = hidden global %swift.async_func_pointer <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @"$s1d3baryySbYaF" to i64), i64 ptrtoint (ptr @"$s1d3baryySbYaFTu" to i64)) to i32), i32 16 }>

define swifttailcc void @"$s1d3fooyySbYaF"(ptr swiftasync %0, i1 %1) {
entry:
  %2 = alloca ptr, align 8
  %c.debug = alloca i1, align 8
  %3 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @"$s1d3fooyySbYaFTu")
  %4 = call ptr @llvm.coro.begin(token %3, ptr null)
  store ptr %0, ptr %2, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %c.debug, i8 0, i64 1, i1 false)
  store i1 %1, ptr %c.debug, align 8
  call void asm sideeffect "", "r"(ptr %c.debug)
  %5 = load i32, ptr getelementptr inbounds (%swift.async_func_pointer, ptr @"$s1d3baryySbYaFTu", i32 0, i32 1), align 8
  %6 = zext i32 %5 to i64
  %7 = call swiftcc ptr @swift_task_alloc(i64 %6) #4
  call void @llvm.lifetime.start.p0(i64 -1, ptr %7)
  %8 = load ptr, ptr %2, align 8
  %9 = getelementptr inbounds <{ ptr, ptr }>, ptr %7, i32 0, i32 0
  store ptr %8, ptr %9, align 8
  %10 = call ptr @llvm.coro.async.resume()
  %11 = getelementptr inbounds <{ ptr, ptr }>, ptr %7, i32 0, i32 1
  store ptr %10, ptr %11, align 8
  %12 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %10, ptr @__swift_async_resume_project_context, ptr @"$s1d3fooyySbYaF.0", ptr @"$s1d3baryySbYaF", ptr %7, i1 %1)
  %13 = extractvalue { ptr } %12, 0
  %14 = call ptr @__swift_async_resume_project_context(ptr %13)
  store ptr %14, ptr %2, align 8
  call swiftcc void @swift_task_dealloc(ptr %7) #4
  call void @llvm.lifetime.end.p0(i64 -1, ptr %7)
  %15 = load ptr, ptr %2, align 8
  %16 = getelementptr inbounds <{ ptr, ptr }>, ptr %15, i32 0, i32 1
  %17 = load ptr, ptr %16, align 8
  %18 = load ptr, ptr %2, align 8
  %19 = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %4, i1 false, ptr @"$s1d3fooyySbYaF.0.1", ptr %17, ptr %18)
  unreachable
}

declare token @llvm.coro.id.async(i32, i32, i32, ptr) #1

declare void @llvm.trap() #2

declare ptr @llvm.coro.begin(token, ptr) #1

declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1 immarg) #3

define hidden swifttailcc void @"$s1d3baryySbYaF"(ptr swiftasync %0, i1 %1) {
entry:
  %2 = alloca ptr, align 8
  %c.debug = alloca i1, align 8
  %3 = call token @llvm.coro.id.async(i32 16, i32 16, i32 0, ptr @"$s1d3baryySbYaFTu")
  %4 = call ptr @llvm.coro.begin(token %3, ptr null)
  store ptr %0, ptr %2, align 8
  call void @llvm.memset.p0.i64(ptr align 8 %c.debug, i8 0, i64 1, i1 false)
  store i1 %1, ptr %c.debug, align 8
  call void asm sideeffect "", "r"(ptr %c.debug)
  br i1 %1, label %5, label %17

5:                                                ; preds = %entry
  %6 = xor i1 %1, true
  %7 = load i32, ptr getelementptr inbounds (%swift.async_func_pointer, ptr @"$s1d3fooyySbYaFTu", i32 0, i32 1), align 8
  %8 = zext i32 %7 to i64
  %9 = call swiftcc ptr @swift_task_alloc(i64 %8) #4
  call void @llvm.lifetime.start.p0(i64 -1, ptr %9)
  %10 = load ptr, ptr %2, align 8
  %11 = getelementptr inbounds <{ ptr, ptr }>, ptr %9, i32 0, i32 0
  store ptr %10, ptr %11, align 8
  %12 = call ptr @llvm.coro.async.resume()
  %13 = getelementptr inbounds <{ ptr, ptr }>, ptr %9, i32 0, i32 1
  store ptr %12, ptr %13, align 8
  %14 = call { ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0s(i32 0, ptr %12, ptr @__swift_async_resume_project_context, ptr @"$s1d3baryySbYaF.0.2", ptr @"$s1d3fooyySbYaF", ptr %9, i1 %6)
  %15 = extractvalue { ptr } %14, 0
  %16 = call ptr @__swift_async_resume_project_context(ptr %15)
  store ptr %16, ptr %2, align 8
  call swiftcc void @swift_task_dealloc(ptr %9) #4
  call void @llvm.lifetime.end.p0(i64 -1, ptr %9)
  br label %18

17:                                               ; preds = %entry
  br label %18

18:                                               ; preds = %5, %17
  %19 = load ptr, ptr %2, align 8
  %20 = getelementptr inbounds <{ ptr, ptr }>, ptr %19, i32 0, i32 1
  %21 = load ptr, ptr %20, align 8
  %22 = load ptr, ptr %2, align 8
  %23 = call i1 (ptr, i1, ...) @llvm.coro.end.async(ptr %4, i1 false, ptr @"$s1d3baryySbYaF.0", ptr %21, ptr %22)
  unreachable
}

declare swiftcc ptr @swift_task_alloc(i64) #4

declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #5

declare ptr @llvm.coro.async.resume() #6

define linkonce_odr hidden ptr @__swift_async_resume_project_context(ptr %0) #7 {
entry:
  %1 = load ptr, ptr %0, align 8
  %2 = call ptr @llvm.swift.async.context.addr()
  store ptr %1, ptr %2, align 8
  ret ptr %1
}

declare ptr @llvm.swift.async.context.addr() #1

define internal swifttailcc void @"$s1d3fooyySbYaF.0"(ptr %0, ptr %1, i1 %2) #8 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1, i1 %2)
  ret void
}

declare { ptr } @llvm.coro.suspend.async.sl_p0s(i32, ptr, ptr, ...) #6

declare swiftcc void @swift_task_dealloc(ptr) #4

declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #5

define internal swifttailcc void @"$s1d3fooyySbYaF.0.1"(ptr %0, ptr %1) #8 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1)
  ret void
}

declare i1 @llvm.coro.end.async(ptr, i1, ...) #1

define internal swifttailcc void @"$s1d3baryySbYaF.0"(ptr %0, ptr %1) #8 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1)
  ret void
}

define internal swifttailcc void @"$s1d3baryySbYaF.0.2"(ptr %0, ptr %1, i1 %2) #8 {
entry:
  musttail call swifttailcc void %0(ptr swiftasync %1, i1 %2)
  ret void
}

attributes #1 = { nounwind }
attributes #2 = { cold noreturn nounwind }
attributes #3 = { nocallback nofree nounwind willreturn}
attributes #4 = { nounwind }
attributes #5 = { nocallback nofree nosync nounwind willreturn }
attributes #6 = { nomerge nounwind }
attributes #7 = { alwaysinline nounwind }
attributes #8 = { alwaysinline nounwind }
