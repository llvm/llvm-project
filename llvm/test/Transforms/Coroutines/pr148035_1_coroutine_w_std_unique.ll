; This is a simple manipulation of the cppreference coroutine example, in llvm IR form, built with async exceptions flag.
; the manipulation is that the coroutine function receives an argument of std::unique_ptr
; which is copied by value
; crashed before fix because of the both validation mismatches:
; "Unwind edges out of a funclet pad must have the same unwind dest"
; and - "Instruction does not dominate all uses!"
; RUN: opt < %s -passes=coro-split -S | FileCheck %s
; CHECK: define

; ModuleID = 'C:\Dev\Projects\clang_asynch_exceptions_coroutines_bug\coroutine_with_unique_ptr_argument.cpp'
source_filename = "C:\\Dev\\Projects\\clang_asynch_exceptions_coroutines_bug\\coroutine_with_unique_ptr_argument.cpp"
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.38.33135"

%struct.awaitable = type { i8 }
%struct.task = type { i8 }
%"class.std::unique_ptr" = type { %"class.std::_Compressed_pair" }
%"class.std::_Compressed_pair" = type { ptr }
%"struct.task::promise_type" = type { i8 }
%"struct.std::suspend_never" = type { i8 }
%"struct.std::_One_then_variadic_args_t" = type { i8 }
%"struct.std::coroutine_handle" = type { ptr }
%"struct.std::coroutine_handle.0" = type { ptr }
%"struct.std::_Zero_then_variadic_args_t" = type { i8 }

$"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@$$QEAV01@@Z" = comdat any

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

$"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ" = comdat any

$"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ" = comdat any

$"?release@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAAPEAHXZ" = comdat any

$"?get_deleter@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAAAEAU?$default_delete@H@2@XZ" = comdat any

$"??$?0U?$default_delete@H@std@@PEAH@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAU?$default_delete@H@1@$$QEAPEAH@Z" = comdat any

$"??$exchange@PEAH$$T@std@@YAPEAHAEAPEAH$$QEA$$T@Z" = comdat any

$"?_Get_first@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAAAEAU?$default_delete@H@2@XZ" = comdat any

$"??0?$coroutine_handle@Upromise_type@task@@@std@@QEAA@XZ" = comdat any

$"?from_address@?$coroutine_handle@X@std@@SA?AU12@QEAX@Z" = comdat any

$"??0?$coroutine_handle@X@std@@QEAA@XZ" = comdat any

$"??R?$default_delete@H@std@@QEBAXPEAH@Z" = comdat any

$"??$?0$$V@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z" = comdat any

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define dso_local i8 @"?switch_to_new_thread@@YA@XZ"() #0 {
  %1 = alloca %struct.awaitable, align 1
  %2 = getelementptr inbounds nuw %struct.awaitable, ptr %1, i32 0, i32 0
  %3 = load i8, ptr %2, align 1
  ret i8 %3
}

; Function Attrs: mustprogress noinline optnone presplitcoroutine sspstrong uwtable
define dso_local i8 @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z"(ptr noundef %0) #1 personality ptr @__CxxFrameHandler3 {
  %2 = alloca %struct.task, align 1
  %3 = alloca ptr, align 8
  %4 = alloca %"class.std::unique_ptr", align 8
  %5 = alloca %"struct.task::promise_type", align 1
  %6 = alloca %"struct.std::suspend_never", align 1
  %7 = alloca %struct.awaitable, align 1
  %8 = alloca %"struct.std::suspend_never", align 1
  store ptr %0, ptr %3, align 8
  invoke void @llvm.seh.scope.begin()
          to label %9 unwind label %107

9:                                                ; preds = %1
  %10 = bitcast ptr %5 to ptr
  %11 = call token @llvm.coro.id(i32 16, ptr %10, ptr null, ptr null)
  %12 = call i1 @llvm.coro.alloc(token %11)
  br i1 %12, label %13, label %17

13:                                               ; preds = %9
  %14 = call i64 @llvm.coro.size.i64()
  %15 = invoke noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %14) #13
          to label %16 unwind label %107

16:                                               ; preds = %13
  br label %17

17:                                               ; preds = %16, %9
  %18 = phi ptr [ null, %9 ], [ %15, %16 ]
  %19 = call ptr @llvm.coro.begin(token %11, ptr %18)
  invoke void @llvm.seh.scope.begin()
          to label %20 unwind label %96

20:                                               ; preds = %17
  call void @llvm.lifetime.start.p0(i64 8, ptr %4) #4
  %21 = call noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %4, ptr noundef nonnull align 8 dereferenceable(8) %0) #4
  invoke void @llvm.seh.scope.begin()
          to label %22 unwind label %91

22:                                               ; preds = %20
  call void @llvm.lifetime.start.p0(i64 1, ptr %5) #4
  invoke void @"?get_return_object@promise_type@task@@QEAA?AU2@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%struct.task) align 1 %2)
          to label %23 unwind label %89

23:                                               ; preds = %22
  invoke void @llvm.seh.scope.begin()
          to label %24 unwind label %85

24:                                               ; preds = %23
  call void @llvm.lifetime.start.p0(i64 1, ptr %6) #4
  invoke void @"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %6)
          to label %25 unwind label %44

25:                                               ; preds = %24
  %26 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %6) #4
  br i1 %26, label %31, label %27

27:                                               ; preds = %25
  %28 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %6, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z.__await_suspend_wrapper__init") #4
  %29 = call i8 @llvm.coro.suspend(token %28, i1 false)
  switch i8 %29, label %83 [
    i8 0, label %31
    i8 1, label %30
  ]

30:                                               ; preds = %27
  br label %32

31:                                               ; preds = %27, %25
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %6) #4
  br label %32

32:                                               ; preds = %31, %30
  %33 = phi i32 [ 0, %31 ], [ 2, %30 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %6) #4
  switch i32 %33, label %73 [
    i32 0, label %34
  ]

34:                                               ; preds = %32
  invoke void @llvm.seh.try.begin()
          to label %35 unwind label %51

35:                                               ; preds = %34
  call void @llvm.lifetime.start.p0(i64 1, ptr %7) #4
  %36 = call i8 @"?switch_to_new_thread@@YA@XZ"()
  %37 = invoke noundef zeroext i1 @"?await_ready@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %7)
          to label %38 unwind label %66

38:                                               ; preds = %35
  br i1 %37, label %46, label %39

39:                                               ; preds = %38
  %40 = call token @llvm.coro.save(ptr null)
  invoke void @llvm.coro.await.suspend.void(ptr %7, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z.__await_suspend_wrapper__await")
          to label %41 unwind label %66

41:                                               ; preds = %39
  %42 = call i8 @llvm.coro.suspend(token %40, i1 false)
  switch i8 %42, label %83 [
    i8 0, label %46
    i8 1, label %43
  ]

43:                                               ; preds = %41
  br label %48

44:                                               ; preds = %24
  %45 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %6) #4
  cleanupret from %45 unwind label %85

46:                                               ; preds = %41, %38
  invoke void @"?await_resume@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %7)
          to label %47 unwind label %66

47:                                               ; preds = %46
  br label %48

48:                                               ; preds = %47, %43
  %49 = phi i32 [ 0, %47 ], [ 2, %43 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %7) #4
  switch i32 %49, label %73 [
    i32 0, label %50
  ]

50:                                               ; preds = %48
  invoke void @"?return_void@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %5)
          to label %65 unwind label %51

51:                                               ; preds = %50, %66, %34
  %52 = catchswitch within none [label %53] unwind label %85

53:                                               ; preds = %51
  %54 = catchpad within %52 [ptr null, i32 0, ptr null]
  invoke void @"?unhandled_exception@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %5) [ "funclet"(token %54) ]
          to label %55 unwind label %85

55:                                               ; preds = %53
  invoke void @llvm.seh.scope.end() [ "funclet"(token %54) ]
          to label %56 unwind label %85

56:                                               ; preds = %55
  catchret from %54 to label %57

57:                                               ; preds = %56
  br label %58

58:                                               ; preds = %57
  br label %59

59:                                               ; preds = %58, %65
  call void @llvm.lifetime.start.p0(i64 1, ptr %8) #4
  call void @"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %5, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %8) #4
  %60 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %8) #4
  br i1 %60, label %68, label %61

61:                                               ; preds = %59
  %62 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %8, ptr %19, ptr @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z.__await_suspend_wrapper__final") #4
  %63 = call i8 @llvm.coro.suspend(token %62, i1 true)
  switch i8 %63, label %83 [
    i8 0, label %68
    i8 1, label %64
  ]

64:                                               ; preds = %61
  br label %69

65:                                               ; preds = %50
  br label %59

66:                                               ; preds = %46, %39, %35
  %67 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %7) #4
  cleanupret from %67 unwind label %51

68:                                               ; preds = %61, %59
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %8) #4
  br label %69

69:                                               ; preds = %68, %64
  %70 = phi i32 [ 0, %68 ], [ 2, %64 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %8) #4
  switch i32 %70, label %73 [
    i32 0, label %71
  ]

71:                                               ; preds = %69
  invoke void @llvm.seh.scope.end()
          to label %72 unwind label %85

72:                                               ; preds = %71
  br label %73

73:                                               ; preds = %72, %69, %48, %32
  %74 = phi i32 [ %33, %32 ], [ %49, %48 ], [ %70, %69 ], [ 0, %72 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %5) #4
  invoke void @llvm.seh.scope.end()
          to label %75 unwind label %91

75:                                               ; preds = %73
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #4
  call void @llvm.lifetime.end.p0(i64 8, ptr %4) #4
  invoke void @llvm.seh.scope.end()
          to label %76 unwind label %96

76:                                               ; preds = %75
  %77 = call ptr @llvm.coro.free(token %11, ptr %19)
  %78 = icmp ne ptr %77, null
  br i1 %78, label %79, label %81

79:                                               ; preds = %76
  %80 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %77, i64 noundef %80) #4
  br label %81

81:                                               ; preds = %76, %79
  switch i32 %74, label %109 [
    i32 0, label %82
    i32 2, label %83
  ]

82:                                               ; preds = %81
  br label %83

83:                                               ; preds = %82, %81, %61, %41, %27
  %84 = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  invoke void @llvm.seh.scope.end()
          to label %104 unwind label %107

85:                                               ; preds = %71, %55, %53, %51, %44, %23
  %86 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %86) ]
          to label %87 unwind label %89

87:                                               ; preds = %85
  %88 = call i1 @llvm.coro.end(ptr null, i1 true, token none) [ "funclet"(token %86) ]
  cleanupret from %86 unwind label %89

89:                                               ; preds = %87, %85, %22
  %90 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %5) #4
  cleanupret from %90 unwind label %91

91:                                               ; preds = %73, %89, %20
  %92 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %92) ]
          to label %93 unwind label %94

93:                                               ; preds = %91
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #4 [ "funclet"(token %92) ]
  cleanupret from %92 unwind label %94

94:                                               ; preds = %93, %91
  %95 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr %4) #4
  cleanupret from %95 unwind label %96

96:                                               ; preds = %75, %94, %17
  %97 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %97) ]
          to label %98 unwind label %107

98:                                               ; preds = %96
  %99 = call ptr @llvm.coro.free(token %11, ptr %19)
  %100 = icmp ne ptr %99, null
  br i1 %100, label %101, label %103

101:                                              ; preds = %98
  %102 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %99, i64 noundef %102) #4 [ "funclet"(token %97) ]
  br label %103

103:                                              ; preds = %98, %101
  cleanupret from %97 unwind label %107

104:                                              ; preds = %83
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4
  %105 = getelementptr inbounds nuw %struct.task, ptr %2, i32 0, i32 0
  %106 = load i8, ptr %105, align 1
  ret i8 %106

107:                                              ; preds = %83, %103, %96, %13, %1
  %108 = cleanuppad within none []
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4 [ "funclet"(token %108) ]
  cleanupret from %108 unwind to caller

109:                                              ; preds = %81
  unreachable
}

declare dso_local i32 @__CxxFrameHandler3(...)

; Function Attrs: nounwind memory(none)
declare dso_local void @llvm.seh.scope.begin() #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: read)
declare token @llvm.coro.id(i32, ptr readnone, ptr nocapture readonly, ptr) #3

; Function Attrs: nounwind
declare i1 @llvm.coro.alloc(token) #4

; Function Attrs: nobuiltin allocsize(0)
declare dso_local noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef) #5

; Function Attrs: nounwind memory(none)
declare i64 @llvm.coro.size.i64() #2

; Function Attrs: nounwind
declare ptr @llvm.coro.begin(token, ptr writeonly) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #6

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca %"struct.std::_One_then_variadic_args_t", align 1
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %7 = load ptr, ptr %4, align 8
  %8 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %7, i32 0, i32 0
  %9 = load ptr, ptr %3, align 8
  %10 = call noundef ptr @"?release@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAAPEAHXZ"(ptr noundef nonnull align 8 dereferenceable(8) %9) #4
  store ptr %10, ptr %5, align 8
  %11 = load ptr, ptr %3, align 8
  %12 = call noundef nonnull align 1 dereferenceable(1) ptr @"?get_deleter@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAAAEAU?$default_delete@H@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %11) #4
  %13 = getelementptr inbounds nuw %"struct.std::_One_then_variadic_args_t", ptr %6, i32 0, i32 0
  %14 = load i8, ptr %13, align 1
  %15 = call noundef ptr @"??$?0U?$default_delete@H@std@@PEAH@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAU?$default_delete@H@1@$$QEAPEAH@Z"(ptr noundef nonnull align 8 dereferenceable(8) %8, i8 %14, ptr noundef nonnull align 1 dereferenceable(1) %12, ptr noundef nonnull align 8 dereferenceable(8) %5) #4
  ret ptr %7
}

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
declare token @llvm.coro.save(ptr) #7

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z.__await_suspend_wrapper__init"(ptr noundef nonnull %0, ptr noundef %1) #8 {
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
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #6

; Function Attrs: nounwind willreturn memory(write)
declare dso_local void @llvm.seh.try.begin() #9

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define internal noundef zeroext i1 @"?await_ready@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %0) #0 align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret i1 false
}

; Function Attrs: alwaysinline mustprogress
define private void @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z.__await_suspend_wrapper__await"(ptr noundef nonnull %0, ptr noundef %1) #8 {
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
define private void @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z.__await_suspend_wrapper__final"(ptr noundef nonnull %0, ptr noundef %1) #8 {
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
define linkonce_odr dso_local void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %5 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %4, i32 0, i32 0
  %6 = load ptr, ptr %5, align 8
  %7 = icmp ne ptr %6, null
  br i1 %7, label %8, label %14

8:                                                ; preds = %1
  %9 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %10 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAAAEAU?$default_delete@H@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %9) #4
  %11 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %12 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %11, i32 0, i32 0
  %13 = load ptr, ptr %12, align 8
  call void @"??R?$default_delete@H@std@@QEBAXPEAH@Z"(ptr noundef nonnull align 1 dereferenceable(1) %10, ptr noundef %13) #4
  br label %14

14:                                               ; preds = %8, %1
  ret void
}

; Function Attrs: nobuiltin nounwind
declare dso_local void @"??3@YAXPEAX_K@Z"(ptr noundef, i64 noundef) #10

; Function Attrs: nounwind memory(argmem: read)
declare ptr @llvm.coro.free(token, ptr nocapture readonly) #11

; Function Attrs: mustprogress noinline norecurse optnone sspstrong uwtable
define dso_local noundef i32 @main() #12 personality ptr @__CxxFrameHandler3 {
  %1 = alloca %"class.std::unique_ptr", align 8
  %2 = alloca %struct.task, align 1
  %3 = call noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4
  invoke void @llvm.seh.scope.begin()
          to label %4 unwind label %7

4:                                                ; preds = %0
  invoke void @llvm.seh.scope.end()
          to label %5 unwind label %7

5:                                                ; preds = %4
  %6 = call i8 @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@@Z"(ptr noundef %1)
  ret i32 0

7:                                                ; preds = %4, %0
  %8 = cleanuppad within none []
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4 [ "funclet"(token %8) ]
  cleanupret from %8 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull returned align 8 dereferenceable(8) %0) unnamed_addr #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca %"struct.std::_Zero_then_variadic_args_t", align 1
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  %5 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %4, i32 0, i32 0
  %6 = getelementptr inbounds nuw %"struct.std::_Zero_then_variadic_args_t", ptr %3, i32 0, i32 0
  %7 = load i8, ptr %6, align 1
  %8 = call noundef ptr @"??$?0$$V@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %5, i8 %7) #4
  ret ptr %4
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"?release@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAAPEAHXZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  %3 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %4 = load ptr, ptr %2, align 8
  store ptr null, ptr %3, align 8
  %5 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %4, i32 0, i32 0
  %6 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %5, i32 0, i32 0
  %7 = call noundef ptr @"??$exchange@PEAH$$T@std@@YAPEAHAEAPEAH$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %3) #4
  ret ptr %7
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @"?get_deleter@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAAAEAU?$default_delete@H@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  %4 = getelementptr inbounds nuw %"class.std::unique_ptr", ptr %3, i32 0, i32 0
  %5 = call noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAAAEAU?$default_delete@H@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %4) #4
  ret ptr %5
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$?0U?$default_delete@H@std@@PEAH@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAU?$default_delete@H@1@$$QEAPEAH@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, i8 %1, ptr noundef nonnull align 1 dereferenceable(1) %2, ptr noundef nonnull align 8 dereferenceable(8) %3) unnamed_addr #0 comdat align 2 {
  %5 = alloca %"struct.std::_One_then_variadic_args_t", align 1
  %6 = alloca ptr, align 8
  %7 = alloca ptr, align 8
  %8 = alloca ptr, align 8
  %9 = getelementptr inbounds nuw %"struct.std::_One_then_variadic_args_t", ptr %5, i32 0, i32 0
  store i8 %1, ptr %9, align 1
  store ptr %3, ptr %6, align 8
  store ptr %2, ptr %7, align 8
  store ptr %0, ptr %8, align 8
  %10 = load ptr, ptr %8, align 8
  %11 = load ptr, ptr %7, align 8
  %12 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %10, i32 0, i32 0
  %13 = load ptr, ptr %6, align 8
  %14 = load ptr, ptr %13, align 8
  store ptr %14, ptr %12, align 8
  ret ptr %10
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$exchange@PEAH$$T@std@@YAPEAHAEAPEAH$$QEA$$T@Z"(ptr noundef nonnull align 8 dereferenceable(8) %0, ptr noundef nonnull align 8 dereferenceable(8) %1) #0 comdat {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = load ptr, ptr %6, align 8
  store ptr %7, ptr %5, align 8
  %8 = load ptr, ptr %3, align 8
  %9 = load ptr, ptr %4, align 8
  store ptr null, ptr %9, align 8
  %10 = load ptr, ptr %5, align 8
  ret ptr %10
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef nonnull align 1 dereferenceable(1) ptr @"?_Get_first@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAAAEAU?$default_delete@H@2@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #0 comdat align 2 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
  %3 = load ptr, ptr %2, align 8
  ret ptr %3
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

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local void @"??R?$default_delete@H@std@@QEBAXPEAH@Z"(ptr noundef nonnull align 1 dereferenceable(1) %0, ptr noundef %1) #0 comdat align 2 personality ptr @__CxxFrameHandler3 {
  %3 = alloca ptr, align 8
  %4 = alloca ptr, align 8
  store ptr %1, ptr %3, align 8
  store ptr %0, ptr %4, align 8
  %5 = load ptr, ptr %4, align 8
  %6 = load ptr, ptr %3, align 8
  %7 = icmp eq ptr %6, null
  br i1 %7, label %11, label %8

8:                                                ; preds = %2
  invoke void @llvm.seh.scope.begin()
          to label %9 unwind label %12

9:                                                ; preds = %8
  invoke void @llvm.seh.scope.end()
          to label %10 unwind label %12

10:                                               ; preds = %9
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 4) #14
  br label %11

11:                                               ; preds = %10, %2
  ret void

12:                                               ; preds = %9, %8
  %13 = cleanuppad within none []
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %6, i64 noundef 4) #14 [ "funclet"(token %13) ]
  cleanupret from %13 unwind to caller
}

; Function Attrs: mustprogress noinline nounwind optnone sspstrong uwtable
define linkonce_odr dso_local noundef ptr @"??$?0$$V@?$_Compressed_pair@U?$default_delete@H@std@@PEAH$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"(ptr noundef nonnull returned align 8 dereferenceable(8) %0, i8 %1) unnamed_addr #0 comdat align 2 {
  %3 = alloca %"struct.std::_Zero_then_variadic_args_t", align 1
  %4 = alloca ptr, align 8
  %5 = getelementptr inbounds nuw %"struct.std::_Zero_then_variadic_args_t", ptr %3, i32 0, i32 0
  store i8 %1, ptr %5, align 1
  store ptr %0, ptr %4, align 8
  %6 = load ptr, ptr %4, align 8
  %7 = getelementptr inbounds nuw %"class.std::_Compressed_pair", ptr %6, i32 0, i32 0
  store ptr null, ptr %7, align 8
  ret ptr %6
}

attributes #0 = { mustprogress noinline nounwind optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { mustprogress noinline optnone presplitcoroutine sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #2 = { nounwind memory(none) }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
attributes #4 = { nounwind }
attributes #5 = { nobuiltin allocsize(0) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #6 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { nomerge nounwind }
attributes #8 = { alwaysinline mustprogress "min-legal-vector-width"="0" }
attributes #9 = { nounwind willreturn memory(write) }
attributes #10 = { nobuiltin nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #11 = { nounwind memory(argmem: read) }
attributes #12 = { mustprogress noinline norecurse optnone sspstrong uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #13 = { allocsize(0) }
attributes #14 = { builtin nounwind }

!llvm.linker.options = !{!0, !1, !2, !3, !4, !5, !6, !7}
!llvm.module.flags = !{!8, !9, !10, !11, !12}
!llvm.ident = !{!13}

!0 = !{!"/DEFAULTLIB:libcmt.lib"}
!1 = !{!"/DEFAULTLIB:oldnames.lib"}
!2 = !{!"/FAILIFMISMATCH:\22_COROUTINE_ABI=2\22"}
!3 = !{!"/FAILIFMISMATCH:\22_MSC_VER=1900\22"}
!4 = !{!"/FAILIFMISMATCH:\22_ITERATOR_DEBUG_LEVEL=0\22"}
!5 = !{!"/FAILIFMISMATCH:\22RuntimeLibrary=MT_StaticRelease\22"}
!6 = !{!"/DEFAULTLIB:libcpmt.lib"}
!7 = !{!"/FAILIFMISMATCH:\22_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\22"}
!8 = !{i32 1, !"wchar_size", i32 2}
!9 = !{i32 2, !"eh-asynch", i32 1}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"uwtable", i32 2}
!12 = !{i32 1, !"MaxTLSAlign", i32 65536}
!13 = !{!"clang version 20.1.6"}
