; RUN: opt < %s -O0 -S

%async_func_ptr = type <{ i32, i32 }>
%Tsq = type <{}>
%swift.context = type { ptr, ptr, i64 }
%swift.type = type { i64 }
%FlatMapSeq = type <{}>
%swift.error = type opaque
%swift.opaque = type opaque

@repoTU = global %async_func_ptr <{ i32 trunc (i64 sub (i64 ptrtoint (ptr @repo to i64), i64 ptrtoint (ptr @repoTU to i64)) to i32), i32 20 }>, section "__TEXT,__const", align 8

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, ptr) #0

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #0

; Function Attrs: nounwind
declare ptr @llvm.coro.async.resume() #0

define hidden ptr @__swift_async_resume_project_context(ptr %0) {
entry:
  ret ptr undef
}

define swifttailcc void @repo(ptr %0, ptr %1, ptr %arg, ptr %2) #1 {
entry:
  %swifterror = alloca swifterror ptr, align 8
  %3 = call token @llvm.coro.id.async(i32 20, i32 16, i32 1, ptr @repoTU)
  %4 = call ptr @llvm.coro.begin(token %3, ptr null)
  %5 = bitcast ptr undef to ptr
  br label %6

6:                                                ; preds = %21, %15, %entry
  br i1 undef, label %7, label %23

7:                                                ; preds = %6
  br i1 undef, label %8, label %16

8:                                                ; preds = %7
  %initializeWithTake35 = bitcast ptr undef to ptr
  %9 = call ptr %initializeWithTake35(ptr noalias %5, ptr noalias undef, ptr undef) #0
  %10 = call ptr @llvm.coro.async.resume()
  %11 = bitcast ptr %10 to ptr
  %12 = call { ptr, ptr } (i32, ptr, ptr, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.error.4.220.413.429.445.461.672.683ss(i32 256, ptr %10, ptr @__swift_async_resume_project_context, ptr @__swift_suspend_dispatch_5.23, ptr undef, ptr undef, ptr undef, ptr %5, ptr undef, ptr undef)
  br i1 undef, label %25, label %13

13:                                               ; preds = %8
  br i1 undef, label %14, label %15

14:                                               ; preds = %13
  br label %24

15:                                               ; preds = %13
  br label %6

16:                                               ; preds = %7
  br i1 undef, label %26, label %17

17:                                               ; preds = %16
  br i1 undef, label %18, label %22

18:                                               ; preds = %17
  br i1 undef, label %27, label %19

19:                                               ; preds = %18
  br i1 undef, label %20, label %21

20:                                               ; preds = %19
  br label %24

21:                                               ; preds = %19
  br label %6

22:                                               ; preds = %17
  br label %24

23:                                               ; preds = %6
  br label %24

24:                                               ; preds = %23, %22, %20, %14
  unreachable

25:                                               ; preds = %8
  br label %28

26:                                               ; preds = %16
  br label %28

27:                                               ; preds = %18
  br label %28

28:                                               ; preds = %27, %26, %25
  unreachable
}

define dso_local swifttailcc void @__swift_suspend_dispatch_2.18() {
entry:
  ret void
}

define dso_local swifttailcc void @__swift_suspend_dispatch_5.19() {
entry:
  ret void
}

define dso_local swifttailcc void @__swift_suspend_dispatch_2.20() {
entry:
  ret void
}

define dso_local swifttailcc void @__swift_suspend_dispatch_4.21() {
entry:
  ret void
}

define dso_local swifttailcc void @__swift_suspend_dispatch_5.22() {
entry:
  ret void
}

define dso_local swifttailcc void @__swift_suspend_dispatch_5.23(ptr %0, ptr %1, ptr %2, ptr %3, ptr %4, ptr %5) {
entry:
  ret void
}

; Function Attrs: nounwind
declare { ptr, ptr } @llvm.coro.suspend.async.sl_p0i8p0s_swift.error.4.220.413.429.445.461.672.683ss(i32, ptr, ptr, ...) #0

attributes #0 = { nounwind }
attributes #1 = { "tune-cpu"="generic" }

!llvm.linker.options = !{!0}

!0 = !{!"-lobjc"}
