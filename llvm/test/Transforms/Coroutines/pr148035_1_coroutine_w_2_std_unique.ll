; This is a simple manipulation of the cppreference coroutine example, in llvm IR form, built with async exceptions flag.
; the manipulation is that the coroutine function receives 2 arguments of std::unique_ptr
; which are copied by value
; crashed before fix because of the both validation mismatches:
; "Unwind edges out of a funclet pad must have the same unwind dest"
; and - "Instruction does not dominate all uses!"
; RUN: opt < %s -passes=coro-split -S --debug | FileCheck %s
; CHECK: define
; ModuleID = 'C:\Dev\Projects\clang_asynch_exceptions_coroutines_bug\coroutine_with_2_unique_ptr_argument.cpp'
source_filename = "C:\\Dev\\Projects\\clang_asynch_exceptions_coroutines_bug\\coroutine_with_2_unique_ptr_argument.cpp"
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
define dso_local i8 @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z"(ptr noundef %0, ptr noundef %1) #1 personality ptr @__CxxFrameHandler3 {
  %3 = alloca %struct.task, align 1
  %4 = alloca ptr, align 8
  %5 = alloca ptr, align 8
  %6 = alloca %"class.std::unique_ptr", align 8
  %7 = alloca %"class.std::unique_ptr", align 8
  %8 = alloca %"struct.task::promise_type", align 1
  %9 = alloca %"struct.std::suspend_never", align 1
  %10 = alloca %struct.awaitable, align 1
  %11 = alloca %"struct.std::suspend_never", align 1
  store ptr %1, ptr %4, align 8
  invoke void @llvm.seh.scope.begin()
          to label %12 unwind label %123

12:                                               ; preds = %2
  store ptr %0, ptr %5, align 8
  invoke void @llvm.seh.scope.begin()
          to label %13 unwind label %117

13:                                               ; preds = %12
  %14 = bitcast ptr %8 to ptr
  %15 = call token @llvm.coro.id(i32 16, ptr %14, ptr null, ptr null)
  %16 = call i1 @llvm.coro.alloc(token %15)
  br i1 %16, label %17, label %21

17:                                               ; preds = %13
  %18 = call i64 @llvm.coro.size.i64()
  %19 = invoke noalias noundef nonnull ptr @"??2@YAPEAX_K@Z"(i64 noundef %18) #13
          to label %20 unwind label %117

20:                                               ; preds = %17
  br label %21

21:                                               ; preds = %20, %13
  %22 = phi ptr [ null, %13 ], [ %19, %20 ]
  %23 = call ptr @llvm.coro.begin(token %15, ptr %22)
  invoke void @llvm.seh.scope.begin()
          to label %24 unwind label %108

24:                                               ; preds = %21
  call void @llvm.lifetime.start.p0(i64 8, ptr %6) #4
  %25 = call noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %6, ptr noundef nonnull align 8 dereferenceable(8) %0) #4
  invoke void @llvm.seh.scope.begin()
          to label %26 unwind label %103

26:                                               ; preds = %24
  call void @llvm.lifetime.start.p0(i64 8, ptr %7) #4
  %27 = call noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@$$QEAV01@@Z"(ptr noundef nonnull align 8 dereferenceable(8) %7, ptr noundef nonnull align 8 dereferenceable(8) %1) #4
  invoke void @llvm.seh.scope.begin()
          to label %28 unwind label %98

28:                                               ; preds = %26
  call void @llvm.lifetime.start.p0(i64 1, ptr %8) #4
  invoke void @"?get_return_object@promise_type@task@@QEAA?AU2@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %8, ptr dead_on_unwind writable sret(%struct.task) align 1 %3)
          to label %29 unwind label %96

29:                                               ; preds = %28
  invoke void @llvm.seh.scope.begin()
          to label %30 unwind label %92

30:                                               ; preds = %29
  call void @llvm.lifetime.start.p0(i64 1, ptr %9) #4
  invoke void @"?initial_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %8, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %9)
          to label %31 unwind label %50

31:                                               ; preds = %30
  %32 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %9) #4
  br i1 %32, label %37, label %33

33:                                               ; preds = %31
  %34 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %9, ptr %23, ptr @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z.__await_suspend_wrapper__init") #4
  %35 = call i8 @llvm.coro.suspend(token %34, i1 false)
  switch i8 %35, label %90 [
    i8 0, label %37
    i8 1, label %36
  ]

36:                                               ; preds = %33
  br label %38

37:                                               ; preds = %33, %31
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %9) #4
  br label %38

38:                                               ; preds = %37, %36
  %39 = phi i32 [ 0, %37 ], [ 2, %36 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %9) #4
  switch i32 %39, label %79 [
    i32 0, label %40
  ]

40:                                               ; preds = %38
  invoke void @llvm.seh.try.begin()
          to label %41 unwind label %57

41:                                               ; preds = %40
  call void @llvm.lifetime.start.p0(i64 1, ptr %10) #4
  %42 = call i8 @"?switch_to_new_thread@@YA@XZ"()
  %43 = invoke noundef zeroext i1 @"?await_ready@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10)
          to label %44 unwind label %72

44:                                               ; preds = %41
  br i1 %43, label %52, label %45

45:                                               ; preds = %44
  %46 = call token @llvm.coro.save(ptr null)
  invoke void @llvm.coro.await.suspend.void(ptr %10, ptr %23, ptr @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z.__await_suspend_wrapper__await")
          to label %47 unwind label %72

47:                                               ; preds = %45
  %48 = call i8 @llvm.coro.suspend(token %46, i1 false)
  switch i8 %48, label %90 [
    i8 0, label %52
    i8 1, label %49
  ]

49:                                               ; preds = %47
  br label %54

50:                                               ; preds = %30
  %51 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %9) #4
  cleanupret from %51 unwind label %92

52:                                               ; preds = %47, %44
  invoke void @"?await_resume@awaitable@?1??switch_to_new_thread@@YA@XZ@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %10)
          to label %53 unwind label %72

53:                                               ; preds = %52
  br label %54

54:                                               ; preds = %53, %49
  %55 = phi i32 [ 0, %53 ], [ 2, %49 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %10) #4
  switch i32 %55, label %79 [
    i32 0, label %56
  ]

56:                                               ; preds = %54
  invoke void @"?return_void@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %8)
          to label %71 unwind label %57

57:                                               ; preds = %56, %72, %40
  %58 = catchswitch within none [label %59] unwind label %92

59:                                               ; preds = %57
  %60 = catchpad within %58 [ptr null, i32 0, ptr null]
  invoke void @"?unhandled_exception@promise_type@task@@QEAAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %8) [ "funclet"(token %60) ]
          to label %61 unwind label %92

61:                                               ; preds = %59
  invoke void @llvm.seh.scope.end() [ "funclet"(token %60) ]
          to label %62 unwind label %92

62:                                               ; preds = %61
  catchret from %60 to label %63

63:                                               ; preds = %62
  br label %64

64:                                               ; preds = %63
  br label %65

65:                                               ; preds = %64, %71
  call void @llvm.lifetime.start.p0(i64 1, ptr %11) #4
  call void @"?final_suspend@promise_type@task@@QEAA?AUsuspend_never@std@@XZ"(ptr noundef nonnull align 1 dereferenceable(1) %8, ptr dead_on_unwind writable sret(%"struct.std::suspend_never") align 1 %11) #4
  %66 = call noundef zeroext i1 @"?await_ready@suspend_never@std@@QEBA_NXZ"(ptr noundef nonnull align 1 dereferenceable(1) %11) #4
  br i1 %66, label %74, label %67

67:                                               ; preds = %65
  %68 = call token @llvm.coro.save(ptr null)
  call void @llvm.coro.await.suspend.void(ptr %11, ptr %23, ptr @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z.__await_suspend_wrapper__final") #4
  %69 = call i8 @llvm.coro.suspend(token %68, i1 true)
  switch i8 %69, label %90 [
    i8 0, label %74
    i8 1, label %70
  ]

70:                                               ; preds = %67
  br label %75

71:                                               ; preds = %56
  br label %65

72:                                               ; preds = %52, %45, %41
  %73 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %10) #4
  cleanupret from %73 unwind label %57

74:                                               ; preds = %67, %65
  call void @"?await_resume@suspend_never@std@@QEBAXXZ"(ptr noundef nonnull align 1 dereferenceable(1) %11) #4
  br label %75

75:                                               ; preds = %74, %70
  %76 = phi i32 [ 0, %74 ], [ 2, %70 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %11) #4
  switch i32 %76, label %79 [
    i32 0, label %77
  ]

77:                                               ; preds = %75
  invoke void @llvm.seh.scope.end()
          to label %78 unwind label %92

78:                                               ; preds = %77
  br label %79

79:                                               ; preds = %78, %75, %54, %38
  %80 = phi i32 [ %39, %38 ], [ %55, %54 ], [ %76, %75 ], [ 0, %78 ]
  call void @llvm.lifetime.end.p0(i64 1, ptr %8) #4
  invoke void @llvm.seh.scope.end()
          to label %81 unwind label %98

81:                                               ; preds = %79
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %7) #4
  call void @llvm.lifetime.end.p0(i64 8, ptr %7) #4
  invoke void @llvm.seh.scope.end()
          to label %82 unwind label %103

82:                                               ; preds = %81
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6) #4
  call void @llvm.lifetime.end.p0(i64 8, ptr %6) #4
  invoke void @llvm.seh.scope.end()
          to label %83 unwind label %108

83:                                               ; preds = %82
  %84 = call ptr @llvm.coro.free(token %15, ptr %23)
  %85 = icmp ne ptr %84, null
  br i1 %85, label %86, label %88

86:                                               ; preds = %83
  %87 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %84, i64 noundef %87) #4
  br label %88

88:                                               ; preds = %83, %86
  switch i32 %80, label %125 [
    i32 0, label %89
    i32 2, label %90
  ]

89:                                               ; preds = %88
  br label %90

90:                                               ; preds = %89, %88, %67, %47, %33
  %91 = call i1 @llvm.coro.end(ptr null, i1 false, token none)
  invoke void @llvm.seh.scope.end()
          to label %116 unwind label %117

92:                                               ; preds = %77, %61, %59, %57, %50, %29
  %93 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %93) ]
          to label %94 unwind label %96

94:                                               ; preds = %92
  %95 = call i1 @llvm.coro.end(ptr null, i1 true, token none) [ "funclet"(token %93) ]
  cleanupret from %93 unwind label %96

96:                                               ; preds = %94, %92, %28
  %97 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 1, ptr %8) #4
  cleanupret from %97 unwind label %98

98:                                               ; preds = %79, %96, %26
  %99 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %99) ]
          to label %100 unwind label %101

100:                                              ; preds = %98
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %7) #4 [ "funclet"(token %99) ]
  cleanupret from %99 unwind label %101

101:                                              ; preds = %100, %98
  %102 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr %7) #4
  cleanupret from %102 unwind label %103

103:                                              ; preds = %81, %101, %24
  %104 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %104) ]
          to label %105 unwind label %106

105:                                              ; preds = %103
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %6) #4 [ "funclet"(token %104) ]
  cleanupret from %104 unwind label %106

106:                                              ; preds = %105, %103
  %107 = cleanuppad within none []
  call void @llvm.lifetime.end.p0(i64 8, ptr %6) #4
  cleanupret from %107 unwind label %108

108:                                              ; preds = %82, %106, %21
  %109 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %109) ]
          to label %110 unwind label %117

110:                                              ; preds = %108
  %111 = call ptr @llvm.coro.free(token %15, ptr %23)
  %112 = icmp ne ptr %111, null
  br i1 %112, label %113, label %115

113:                                              ; preds = %110
  %114 = call i64 @llvm.coro.size.i64()
  call void @"??3@YAXPEAX_K@Z"(ptr noundef %111, i64 noundef %114) #4 [ "funclet"(token %109) ]
  br label %115

115:                                              ; preds = %110, %113
  cleanupret from %109 unwind label %117

116:                                              ; preds = %90
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4
  invoke void @llvm.seh.scope.end()
          to label %120 unwind label %123

117:                                              ; preds = %90, %115, %108, %17, %12
  %118 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %118) ]
          to label %119 unwind label %123

119:                                              ; preds = %117
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %0) #4 [ "funclet"(token %118) ]
  cleanupret from %118 unwind label %123

120:                                              ; preds = %116
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4
  %121 = getelementptr inbounds nuw %struct.task, ptr %3, i32 0, i32 0
  %122 = load i8, ptr %121, align 1
  ret i8 %122

123:                                              ; preds = %116, %119, %117, %2
  %124 = cleanuppad within none []
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4 [ "funclet"(token %124) ]
  cleanupret from %124 unwind to caller

125:                                              ; preds = %88
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
define private void @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z.__await_suspend_wrapper__init"(ptr noundef nonnull %0, ptr noundef %1) #8 {
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
define private void @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z.__await_suspend_wrapper__await"(ptr noundef nonnull %0, ptr noundef %1) #8 {
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
define private void @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z.__await_suspend_wrapper__final"(ptr noundef nonnull %0, ptr noundef %1) #8 {
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
  %2 = alloca %"class.std::unique_ptr", align 8
  %3 = alloca %struct.task, align 1
  %4 = call noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4
  invoke void @llvm.seh.scope.begin()
          to label %5 unwind label %14

5:                                                ; preds = %0
  %6 = call noundef ptr @"??$?0U?$default_delete@H@std@@$0A@@?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %2) #4
  invoke void @llvm.seh.scope.begin()
          to label %7 unwind label %11

7:                                                ; preds = %5
  invoke void @llvm.seh.scope.end()
          to label %8 unwind label %11

8:                                                ; preds = %7
  invoke void @llvm.seh.scope.end()
          to label %9 unwind label %14

9:                                                ; preds = %8
  %10 = call i8 @"?resuming_on_new_thread@@YA?AUtask@@V?$unique_ptr@HU?$default_delete@H@std@@@std@@0@Z"(ptr noundef %2, ptr noundef %1)
  ret i32 0

11:                                               ; preds = %7, %5
  %12 = cleanuppad within none []
  invoke void @llvm.seh.scope.end() [ "funclet"(token %12) ]
          to label %13 unwind label %14

13:                                               ; preds = %11
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %2) #4 [ "funclet"(token %12) ]
  cleanupret from %12 unwind label %14

14:                                               ; preds = %8, %13, %11, %0
  %15 = cleanuppad within none []
  call void @"??1?$unique_ptr@HU?$default_delete@H@std@@@std@@QEAA@XZ"(ptr noundef nonnull align 8 dereferenceable(8) %1) #4 [ "funclet"(token %15) ]
  cleanupret from %15 unwind to caller
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
