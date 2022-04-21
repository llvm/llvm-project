; RUN: opt < %s -O0 -S

%async_func_ptr = type <{ i32, i32 }>
%Tsq = type <{}>
%swift.context = type { %swift.context*, void (%swift.context*)*, i64 }
%swift.type = type { i64 }
%FlatMapSeq = type <{}>
%swift.error = type opaque
%swift.opaque = type opaque

@repoTU = global %async_func_ptr <{ i32 trunc (i64 sub (i64 ptrtoint (void (%Tsq*, %swift.context*, %swift.type*, %FlatMapSeq*)* @repo to i64), i64 ptrtoint (%async_func_ptr* @repoTU to i64)) to i32), i32 20 }>, section "__TEXT,__const", align 8

; Function Attrs: nounwind
declare token @llvm.coro.id.async(i32, i32, i32, i8*) #0

; Function Attrs: nounwind
declare i8* @llvm.coro.begin(token, i8* writeonly) #0

; Function Attrs: nounwind
declare i8* @llvm.coro.async.resume() #0

define hidden i8* @__swift_async_resume_project_context(i8* %0) {
entry:
  ret i8* undef
}

define swifttailcc void @repo(%Tsq* %0, %swift.context* %1, %swift.type* %arg, %FlatMapSeq* %2) #1 {
entry:
  %swifterror = alloca swifterror %swift.error*, align 8
  %3 = call token @llvm.coro.id.async(i32 20, i32 16, i32 1, i8* bitcast (%async_func_ptr* @repoTU to i8*))
  %4 = call i8* @llvm.coro.begin(token %3, i8* null)
  %5 = bitcast i8* undef to %swift.opaque*
  br label %6

6:                                                ; preds = %21, %15, %entry
  br i1 undef, label %7, label %23

7:                                                ; preds = %6
  br i1 undef, label %8, label %16

8:                                                ; preds = %7
  %initializeWithTake35 = bitcast i8* undef to %swift.opaque* (%swift.opaque*, %swift.opaque*, %swift.type*)*
  %9 = call %swift.opaque* %initializeWithTake35(%swift.opaque* noalias %5, %swift.opaque* noalias undef, %swift.type* undef) #0
  %10 = call i8* @llvm.coro.async.resume()
  %11 = bitcast i8* %10 to void (%swift.context*)*
  %12 = call { i8*, %swift.error* } (i32, i8*, i8*, ...) @llvm.coro.suspend.async.sl_p0i8p0s_swift.error.4.220.413.429.445.461.672.683ss(i32 256, i8* %10, i8* bitcast (i8* (i8*)* @__swift_async_resume_project_context to i8*), i8* bitcast (void (i8*, %Tsq*, %swift.context*, %swift.opaque*, %swift.type*, i8**)* @__swift_suspend_dispatch_5.23 to i8*), i8* undef, %Tsq* undef, %swift.context* undef, %swift.opaque* %5, %swift.type* undef, i8** undef)
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

define dso_local swifttailcc void @__swift_suspend_dispatch_5.23(i8* %0, %Tsq* %1, %swift.context* %2, %swift.opaque* %3, %swift.type* %4, i8** %5) {
entry:
  ret void
}

; Function Attrs: nounwind
declare { i8*, %swift.error* } @llvm.coro.suspend.async.sl_p0i8p0s_swift.error.4.220.413.429.445.461.672.683ss(i32, i8*, i8*, ...) #0

attributes #0 = { nounwind }
attributes #1 = { "tune-cpu"="generic" }

!llvm.linker.options = !{!0}

!0 = !{!"-lobjc"}
