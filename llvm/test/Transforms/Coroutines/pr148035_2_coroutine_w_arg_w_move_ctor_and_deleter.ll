; This is a simple manipulation of the cppreference coroutine example, in llvm IR form, built with async exceptions flag.
; the manipulation is that the coroutine function receives an argument with is of class type
; that's having a default move ctor, and a dtor which tries to delete a member pointer.
; It's a cpp reduce of std::unique_ptr
; crashed before fix because of the both validation mismatches:
; "Unwind edges out of a funclet pad must have the same unwind dest"
; and - "Instruction does not dominate all uses!"
; RUN: opt < %s -passes=coro-split -S | FileCheck %s
; CHECK: define

; ModuleID = 'coroutine_with_argument_having_deleter_and_move_ctor.cpp'
source_filename = "coroutine_with_argument_having_deleter_and_move_ctor.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.38.33135"

%struct.awaitable = type { i8 }
%struct.task = type { i8 }
%class.unique_ptr = type { ptr }
%"struct.task::promise_type" = type { i8 }
%"struct.std::suspend_never" = type { i8 }
%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { ptr }

$"?get_return_object@promise_type@task@@QEAA?AU2@XZ" = comdat any

$"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ" = comdat any

$"?await_ready@suspend_never@std@@QEBA_NXZ" = comdat any

$"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z" = comdat any

$"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z" = comdat any

$"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ" = comdat any

$"?await_resume@suspend_never@std@@QEBAXXZ" = comdat any

$"?return_void@promise_type@task@@QEAAXXZ" = comdat any

$"?unhandled_exception@promise_type@task@@QEAAXXZ" = comdat any

$"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ" = comdat any

$"??1unique_ptr@@QEAA@XZ" = comdat any

$"??0?$coroutine_handle@Upromise_type@task@@@std@@QEAA@XZ" = comdat any

$"?from_address@?$coroutine_handle@X@std@@SA?AU12@QEAX@Z" = comdat any

$"??0?$coroutine_handle@X@std@@QEAA@XZ" = comdat any

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define dso_local i8 @"?switch_to_new_thread@@YA@XZ"() #0 {
  %1 = alloca %struct.awaitable, align 1
  %2 = getelementptr inbounds nuw %struct.awaitable, ptr %1, i32 0, i32 0
  %3 = load i8, ptr %2, align 1
  ret i8 %3
}

; Function Attrs: mustprogress noinline optnone presplitcoroutine sspstrong uwtable
define dso_local i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(ptr noundef %0) #1 personality ptr @__CxxFrameHandler3 {
  %2 = alloca %struct.task, align 1
  %3 = alloca ptr, align 8
  %4 = alloca %class.unique_ptr, align 8
  %5 = alloca %"struct.task::promise_type", align 1
  %6 = alloca %"struct.std::suspend_never", align 1
  %7 = alloca %struct.awaitable, align 1
  %8 = alloca %"struct.std::suspend_never", align 1
  store ptr %0, ptr %3, align 8
  invoke void @llvm.seh.scope.begin()
          to label %9 unwind label %106

9:                                                ; preds = %1
  %10 = bitcast ptr %5 to ptr
  %11 = call token @llvm.coro.id(i32 16, ptr %10, ptr null, ptr null)
  %12 = call i1 @llvm.coro.alloc(token %11)
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = call i64 @llvm.coro.size.i64()
  %15 = invoke noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %14) #14
          to label %16 unwind label %106

16:                                               ; preds = %13
  br label %17

17:                                               ; preds = %16, %9
  %18 = phi ptr [ null, %9 ], [ %15, %16 ]
  %19 = call ptr @llvm.coro.begin(token %11, ptr %18)
  invoke void @llvm.seh.scope.begin()
          to label %20 unwind label %95

20:                                               ; preds = %17
  call void @llvm.lifetime.start.p0(i64 8, ptr %4) #4
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %4, ptr align 8 %0, i64 8, i1 false)
  invoke void @llvm.seh.scope.begin()
          to label %21 unwind label %90

21:                                               ; preds = %20
  call void @llvm.lifetime.start.p0(i64 1, ptr %5) #4
  invoke void @"?get_return_object@promise_type@task@@QEAA?AU2@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%struct.task) align 1 %2)
          to label %22 unwind label %88

22:                                               ; preds = %21
  invoke void @llvm.seh.scope.begin()
          to label %23 unwind label %84

23:                                               ; preds = %22
  call void @llvm.lifetime.start.p0(i64 1, ptr %6) #4
  invoke void @"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %6)
          to label %24 unwind label %43

24:                                               ; preds = %23
  %25 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %6) #4
  br i1 %25, label %30, label %26

26:                                               ; preds = %24
  %27 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %6, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z.__await_suspend_wrapper__init") #4
  %28 = call i8 @llvm.coro.suspend(token %27, i1 false)
  switch i8 %28, label %82 [
    i8 0, label %30
    i8 1, label %29
  ]

29:                                               ; preds = %26
  br label %31

30:                                               ; preds = %26, %24
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %6) #4
  br label %31

31:                                               ; preds = %30, %29
  %32 = phi i32 [ 0, %30 ], [ 2, %29 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %6) #4
  switch i32 %32, label %72 [
    i32 0, label %33
  ]

33:                                               ; preds = %31
  invoke void @llvm.seh.try.begin()
          to label %34 unwind label %50

34:                                               ; preds = %33
  call void @llvm.lifetime.start.p0(i64 1, ptr %7) #4
  %35 = call i8 @"?switch_to_new_thread@@YA@XZ"()
  %36 = invoke noundef zeroext i1 @"?await_ready@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %7)
          to label %37 unwind label %65

37:                                               ; preds = %34
  br i1 %36, label %45, label %38

38:                                               ; preds = %37
  %39 = call token @llvm.coro.save(ptr null)
  invoke void @llvm.coro.await.suspend.void(ptr %7, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z.__await_suspend_wrapper__await")
          to label %40 unwind label %65

40:                                               ; preds = %38
  %41 = call i8 @llvm.coro.suspend(token %39, i1 false)
  switch i8 %41, label %82 [
    i8 0, label %45
    i8 1, label %42
  ]

42:                                               ; preds = %40
  br label %47

43:                                               ; preds = %23
  %44 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %6) #4
  cleanupret from %44 unwind label %84

45:                                               ; preds = %40, %37
  invoke void @"?await_resume@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %7)
          to label %46 unwind label %65

46:                                               ; preds = %45
  br label %47

47:                                               ; preds = %46, %42
  %48 = phi i32 [ 0, %46 ], [ 2, %42 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %7) #4
  switch i32 %48, label %72 [
    i32 0, label %49
  ]

49:                                               ; preds = %47
  invoke void @"?return_void@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %5)
          to label %64 unwind label %50

50:                                               ; preds = %49, %65, %33
  %51 = catchswitch within none [label %52] unwind label %84

52:                                               ; preds = %50
  %53 = catchpad within %51 [ptr null, i32 0, ptr null]
  invoke void @"?unhandled_exception@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %5) [ "funclet"(token %53) ]
          to label %54 unwind label %84

54:                                               ; preds = %52
  invoke void @llvm.seh.scope.end() [ "funclet"(token %53) ]
          to label %55 unwind label %84

55:                                               ; preds = %54
  catchret from %53 to label %56

56:                                               ; preds = %55
  br label %57

57:                                               ; preds = %56
  br label %58

58:                                               ; preds = %57, %64
  call void @llvm.lifetime.start.p0(i64 1, ptr %8) #4
  call void @"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %8) #4
  %59 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %8) #4
  br i1 %59, label %67, label %60

60:                                               ; preds = %58
  %61 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %8, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z.__await_suspend_wrapper__final") #4
  %62 = call i8 @llvm.coro.suspend(token %61, i1 true)
  switch i8 %62, label %82 [
    i8 0, label %67
    i8 1, label %63
  ]

63:                                               ; preds = %60
  br label %68

64:                                               ; preds = %49
  br label %58

65:                                               ; preds = %45, %38, %34
  %66 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %7) #4
  cleanupret from %66 unwind label %50

67:                                               ; preds = %60, %58
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %8) #4
  br label %68

68:                                               ; preds = %67, %63
  %69 = phi i32 [ 0, %67 ], [ 2, %63 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %8) #4
  switch i32 %69, label %72 [
    i32 0, label %70
  ]

70:                                               ; preds = %68
  invoke void @llvm.seh.scope.end()
          to label %71 unwind label %84

71:                                               ; preds = %70
  br label %72

72:                                               ; preds = %71, %68, %47, %31
  %73 = phi i32 [ %32, %31 ], [ %48, %47 ], [ %69, %68 ], [ 0, %71 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %5) #4
  invoke void @llvm.seh.scope.end()
          to label %74 unwind label %90

74:                                               ; preds = %72
  call void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #4
  call void @llvm.lifetime.end.p0(i64 8, ptr %4) #4
  invoke void @llvm.seh.scope.end()
          to label %75 unwind label %95

75:                                               ; preds = %74
  %76 = call ptr @llvm.coro.free(token %11, ptr %19)
  %77 = icmp ne ptr %76, null
  br i1 %77, label %78, label %80

78:                                               ; preds = %75
  %79 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %76, i64 noundef %79) #4
  br label %80

80:                                               ; preds = %75, %78
  switch i32 %73, label %108 [
    i32 0, label %81
    i32 2, label %82
  ]

81:                                               ; preds = %80
  br label %82

82:                                               ; preds = %81, %80, %60, %40, %26
  %83 = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  invoke void @llvm.seh.scope.end()
          to label %103 unwind label %106

84:                                               ; preds = %70, %54, %52, %50, %43, %22
  %85 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %85) ]
          to label %86 unwind label %88

86:                                               ; preds = %84
  %87 = call i1 @llvm.coro.end(ptr null, i1 true, token none) [ "funclet"(token %85) ]
  cleanupret from %85 unwind label %88

88:                                               ; preds = %86, %84, %21
  %89 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %5) #4
  cleanupret from %89 unwind label %90

90:                                               ; preds = %72, %88, %20
  %91 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %91) ]
          to label %92 unwind label %93

92:                                               ; preds = %90
  call void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #4 [ "funclet"(token %91) ]
  cleanupret from %91 unwind label %93

93:                                               ; preds = %92, %90
  %94 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr %4) #4
  cleanupret from %94 unwind label %95

95:                                               ; preds = %74, %93, %17
  %96 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %96) ]
          to label %97 unwind label %106

97:                                               ; preds = %95
  %98 = call ptr @llvm.coro.free(token %11, ptr %19)
  %99 = icmp ne ptr %98, null
  br i1 %99, label %100, label %102

100:                                              ; preds = %97
  %101 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %98, i64 noundef %101) #4 [ "funclet"(token %96) ]
  br label %102

102:                                              ; preds = %97, %100
  cleanupret from %96 unwind label %106

103:                                              ; preds = %82
  call void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4
  %104 = getelementptr inbounds nuw %struct.task, ptr %2, i32 0, i32 0
  %105 = load i8, ptr %104, align 1
  ret i8 %105

106:                                              ; preds = %82, %102, %95, %13, %1
  %107 = cleanuppad within none []
  call void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4 [ "funclet"(token %107) ]
  cleanupret from %107 unwind to caller

108:                                              ; preds = %80
  unreachable
}

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind memory(none)
declare dso_local void @llvm.seh.scope.begin() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr readonly captures(none), ptr) #3

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #4

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef) #5

; Function Attrs: nounwind memory(none)
declare i64 @llvm.coro.size.i64() #2

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #6

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #7

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?get_return_object@promise_type@task@@QEAA?AU2@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr dead_on_unwind noalias writable sret(%struct.task) align 1 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr dead_on_unwind noalias writable sret(%"struct.std::suspend_never") align 1 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret i1 true
}

; Function Attrs: nomerge nounwind
declare token @llvm.coro.save(ptr) #8

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z.__await_suspend_wrapper__init"(ptr noundef nonnull %0, ptr noundef %1) #9 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::coroutine_handle", align 8
  %6 = alloca %"struct.std::coroutine_handle.0", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %3, align 8
  call void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle.0") align 8 %6, ptr noundef %8) #4
  call void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %5) #4
  %9 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %5, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %10 to i64
  call void @"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %7, i64 %11) #4
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, i64 %1) #0 comdat align 2 {
  %3 = alloca %"struct.std::coroutine_handle", align 8
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %3, i32 0, i32 0
  %6 = inttoptr i64 %1 to ptr
  store ptr %6, ptr %5, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind noalias writable sret(%"struct.std::coroutine_handle.0") align 8 %0, ptr noundef %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = call noundef ptr @"??0?$coroutine_handle@Upromise_type@task@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"struct.std::coroutine_handle.0", ptr %0, i32 0, i32 0
  store ptr %6, ptr %7, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr dead_on_unwind noalias writable sret(%"struct.std::coroutine_handle") align 8 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = getelementptr inbounds nuw %"struct.std::coroutine_handle.0", ptr %5, i32 0, i32 0
  %7 = load ptr, ptr %6, align 8
  call void @"?from_address@?$coroutine_handle@X@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %1, ptr noundef %7) #4
  ret void
}

declare void @llvm.coro.await.suspend.void(ptr, ptr, ptr)

; Function Attrs: nounwind
declare i8 @llvm.coro.suspend(token, i1) #4

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #6

; Function Attrs: nounwind willreturn memory(write)
declare dso_local void @llvm.seh.try.begin() #10

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef zeroext i1 @"?await_ready@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret i1 false
}

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z.__await_suspend_wrapper__await"(ptr noundef nonnull %0, ptr noundef %1) #9 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::coroutine_handle", align 8
  %6 = alloca %"struct.std::coroutine_handle.0", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %3, align 8
  call void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle.0") align 8 %6, ptr noundef %8) #4
  call void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %5) #4
  %9 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %5, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %10 to i64
  call void @"?await_suspend@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAAXU?$coroutine_handle@X@std@@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %7, i64 %11)
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal void @"?await_suspend@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAAXU?$coroutine_handle@X@std@@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, i64 %1) #0 align 2 {
  %3 = alloca %"struct.std::coroutine_handle", align 8
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %3, i32 0, i32 0
  %6 = inttoptr i64 %1 to ptr
  store ptr %6, ptr %5, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal void @"?await_resume@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?return_void@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?unhandled_exception@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret void
}

; Function Attrs: nounwind memory(none)
declare dso_local void @llvm.seh.scope.end() #2

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr dead_on_unwind noalias writable sret(%"struct.std::suspend_never") align 1 %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  ret void
}

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z.__await_suspend_wrapper__final"(ptr noundef nonnull %0, ptr noundef %1) #9 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca %"struct.std::coroutine_handle", align 8
  %6 = alloca %"struct.std::coroutine_handle.0", align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = load ptr, ptr %3, align 8
  call void @"?from_address@?$coroutine_handle@Upromise_type@task@@@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle.0") align 8 %6, ptr noundef %8) #4
  call void @"??B?$coroutine_handle@Upromise_type@task@@@std@@QEBA?AU?$coroutine_handle@X@1@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr dead_on_unwind writable sret(%"struct.std::coroutine_handle") align 8 %5) #4
  %9 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %5, i32 0, i32 0
  %10 = load ptr, ptr %9, align 8
  %11 = ptrtoint ptr %10 to i64
  call void @"?await_suspend@suspend_never@std@@QEBAXU?$coroutine_handle@X@2@@Z"(ptr noundef nonnull align 1 dereferenceable(1) %7, i64 %11) #4
  ret void
}

; Function Attrs: nounwind
declare i1 @llvm.coro.end(ptr, i1, token) #4

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %class.unique_ptr, ptr %3, i32 0, i32 0
  %5 = load ptr, ptr %4, align 8
  %6 = icmp eq ptr %5, null
  br i1 %6, label %10, label %7

7:                                                ; preds = %1
  invoke void @llvm.seh.scope.begin()
          to label %8 unwind label %11

8:                                                ; preds = %7
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %11

9:                                                ; preds = %8
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %5, i64 noundef 4) #15
  br label %10

10:                                               ; preds = %9, %1
  ret void

11:                                               ; preds = %8, %7
  %12 = cleanuppad within none []
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %5, i64 noundef 4) #15 [ "funclet"(token %12) ]
  cleanupret from %12 unwind to caller
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @"??3@YAXPEAX_K@Z"(ptr noundef, i64 noundef) #11

; Function Attrs: nounwind memory(argmem: read)
declare ptr @llvm.coro.free(token, ptr readonly captures(none)) #12

; Function Attrs: mustprogress noinline norecurse optnone sspstrong uwtable
define dso_local noundef i32 @main() #13 personality ptr @__CxxFrameHandler3 {
  %1 = alloca %class.unique_ptr, align 8
  %2 = alloca %class.unique_ptr, align 8
  %3 = alloca %struct.task, align 1
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %13

4:                                                ; preds = %0
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %2, ptr align 8 %1, i64 8, i1 false)
  invoke void @llvm.seh.scope.begin()
          to label %5 unwind label %10

5:                                                ; preds = %4
  invoke void @llvm.seh.scope.end()
          to label %6 unwind label %10

6:                                                ; preds = %5
  %7 = invoke i8 @"?resuming_on_new_thread@@YA?AUtask@@Vunique_ptr@@@Z"(ptr noundef %2)
          to label %8 unwind label %13

8:                                                ; preds = %6
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %13

9:                                                ; preds = %8
  call void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4
  ret i32 0

10:                                               ; preds = %5, %4
  %11 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %11) ]
          to label %12 unwind label %13

12:                                               ; preds = %10
  call void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %2) #4 [ "funclet"(token %11) ]
  cleanupret from %11 unwind label %13

13:                                               ; preds = %8, %6, %12, %10, %0
  %14 = cleanuppad within none []
  call void @"??1unique_ptr@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4 [ "funclet"(token %14) ]
  cleanupret from %14 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$coroutine_handle@Upromise_type@task@@@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::coroutine_handle.0", ptr %3, i32 0, i32 0
  store ptr null, ptr %4, align 8
  ret ptr %3
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"?from_address@?$coroutine_handle@X@std@@SA?AU12@QEAX@Z"(ptr dead_on_unwind noalias writable sret(%"struct.std::coroutine_handle") align 8 %0, ptr noundef %1) #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %0, ptr %3, align 8
  store ptr %1, ptr %4, align 8
  %5 = call noundef ptr @"??0?$coroutine_handle@X@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %0, i32 0, i32 0
  store ptr %6, ptr %7, align 8
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??0?$coroutine_handle@X@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"struct.std::coroutine_handle", ptr %3, i32 0, i32 0
  store ptr null, ptr %4, align 8
  ret ptr %3
}

attributes #0 = { mustprogress noinline nounwind optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline optnone presplitcoroutine sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { nounwind }
attributes #5 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { nomerge nounwind }
attributes #9 = { alwaysinline mustprogress "min-legal-vector-width"="0" }
attributes #10 = { nounwind willreturn memory(write) }
attributes #11 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #12 = { nounwind memory(argmem: read) }
attributes #13 = { mustprogress noinline norecurse optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #14 = { allocsize(0) }
attributes #15 = { builtin nounwind }

!llvm.linker.options = !{!0, !1, !2}
!llvm.module.flags = !{!3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = !{!"/DEFAULTLIB:libcmt.lib"}
!1 = !{!"/DEFAULTLIB:oldnames.lib"}
!2 = !{!"/FAILIFMISMATCH:\22_COROUTINE_ABI=2\22"}
!3 = !{i32 1, !"wchar_size", i32 2}
!4 = !{i32 2, !"eh-asynch", i32 1}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"uwtable", i32 2}
!7 = !{i32 1, !"MaxTLSAlign", i32 65536}
!8 = !{!"clang version 21.0.0git (https://github.com/llvm/llvm-project.git e66c205bda33a91fbe2ba5b4a5d6b823e5c23e8a)"}
