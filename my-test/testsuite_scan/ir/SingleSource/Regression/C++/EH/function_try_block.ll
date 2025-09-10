; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/function_try_block.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/function_try_block.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.B = type { %struct.A, %struct.A, %struct.A, i32, %struct.A, %struct.A }
%struct.A = type { i32 }

$__clang_call_terminate = comdat any

@_ZL11ShouldThrow = internal unnamed_addr global i1 false, align 1
@_ZTIi = external constant ptr
@.str.2 = private unnamed_addr constant [47 x i8] c"In B catch block with int %d: auto rethrowing\0A\00", align 1
@_ZL5NumAs = internal unnamed_addr global i32 0, align 4
@.str.4 = private unnamed_addr constant [15 x i8] c"Created A #%d\0A\00", align 1
@.str.5 = private unnamed_addr constant [17 x i8] c"Destroyed A #%d\0A\00", align 1
@str = private unnamed_addr constant [41 x i8] c"'throws' threw an exception: rethrowing!\00", align 4
@str.7 = private unnamed_addr constant [18 x i8] c"In B constructor!\00", align 4
@str.8 = private unnamed_addr constant [18 x i8] c"Caught exception!\00", align 4
@str.9 = private unnamed_addr constant [14 x i8] c"B destructor!\00", align 4

@_ZN1BC1Ev = dso_local unnamed_addr alias void (ptr), ptr @_ZN1BC2Ev

; Function Attrs: mustprogress uwtable
define dso_local noundef i32 @_Z6throwsv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = load i1, ptr @_ZL11ShouldThrow, align 1
  br i1 %1, label %2, label %9

2:                                                ; preds = %0
  %3 = tail call ptr @__cxa_allocate_exception(i64 4) #9
  store i32 7, ptr %3, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %3, ptr nonnull @_ZTIi, ptr null) #10
          to label %16 unwind label %4

4:                                                ; preds = %2
  %5 = landingpad { ptr, i32 }
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = tail call ptr @__cxa_begin_catch(ptr %6) #9
  %8 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  invoke void @__cxa_rethrow() #10
          to label %16 unwind label %10

9:                                                ; preds = %0
  ret i32 123

10:                                               ; preds = %4
  %11 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %12 unwind label %13

12:                                               ; preds = %10
  resume { ptr, i32 } %11

13:                                               ; preds = %10
  %14 = landingpad { ptr, i32 }
          catch ptr null
  %15 = extractvalue { ptr, i32 } %14, 0
  tail call void @__clang_call_terminate(ptr %15) #11
  unreachable

16:                                               ; preds = %4, %2
  unreachable
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #1

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

declare void @__cxa_rethrow() local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #3 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #9
  tail call void @_ZSt9terminatev() #11
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #4

; Function Attrs: mustprogress uwtable
define dso_local void @_ZN1BC2Ev(ptr noundef nonnull align 4 captures(none) dereferenceable(24) initializes((0, 12)) %0) unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %2 = load i32, ptr @_ZL5NumAs, align 4, !tbaa !6
  %3 = add i32 %2, 1
  store i32 %3, ptr @_ZL5NumAs, align 4, !tbaa !6
  store i32 %2, ptr %0, align 4, !tbaa !10
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %2)
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %6 = load i32, ptr @_ZL5NumAs, align 4, !tbaa !6
  %7 = add i32 %6, 1
  store i32 %7, ptr @_ZL5NumAs, align 4, !tbaa !6
  store i32 %6, ptr %5, align 4, !tbaa !10
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %6)
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %10 = load i32, ptr @_ZL5NumAs, align 4, !tbaa !6
  %11 = add i32 %10, 1
  store i32 %11, ptr @_ZL5NumAs, align 4, !tbaa !6
  store i32 %10, ptr %9, align 4, !tbaa !10
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %10)
  %13 = invoke noundef i32 @_Z6throwsv()
          to label %14 unwind label %25

14:                                               ; preds = %1
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 12
  store i32 123, ptr %15, align 4, !tbaa !12
  %16 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %17 = load i32, ptr @_ZL5NumAs, align 4, !tbaa !6
  %18 = add i32 %17, 1
  store i32 %18, ptr @_ZL5NumAs, align 4, !tbaa !6
  store i32 %17, ptr %16, align 4, !tbaa !10
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %17)
  %20 = getelementptr inbounds nuw i8, ptr %0, i64 20
  %21 = load i32, ptr @_ZL5NumAs, align 4, !tbaa !6
  %22 = add i32 %21, 1
  store i32 %22, ptr @_ZL5NumAs, align 4, !tbaa !6
  store i32 %21, ptr %20, align 4, !tbaa !10
  %23 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %21)
  %24 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  ret void

25:                                               ; preds = %1
  %26 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIi
  %27 = load i32, ptr %9, align 4, !tbaa !10
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %27)
  %29 = load i32, ptr %5, align 4, !tbaa !10
  %30 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %29)
  %31 = load i32, ptr %0, align 4, !tbaa !10
  %32 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %31)
  %33 = extractvalue { ptr, i32 } %26, 1
  %34 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #9
  %35 = icmp eq i32 %33, %34
  br i1 %35, label %36, label %44

36:                                               ; preds = %25
  %37 = extractvalue { ptr, i32 } %26, 0
  %38 = tail call ptr @__cxa_begin_catch(ptr %37) #9
  %39 = load i32, ptr %38, align 4, !tbaa !6
  %40 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %39)
  invoke void @__cxa_rethrow()
          to label %41 unwind label %42

41:                                               ; preds = %36
  unreachable

42:                                               ; preds = %36
  %43 = landingpad { ptr, i32 }
          cleanup
  tail call void @__cxa_end_catch() #9
  br label %44

44:                                               ; preds = %42, %25
  %45 = phi { ptr, i32 } [ %43, %42 ], [ %26, %25 ]
  resume { ptr, i32 } %45
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #6

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #6

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #7 personality ptr @__gxx_personality_v0 {
  %1 = alloca %struct.B, align 4
  %2 = alloca %struct.B, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #9
  call void @_ZN1BC2Ev(ptr noundef nonnull align 4 dereferenceable(24) %1)
  %3 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.9)
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %5 = load i32, ptr %4, align 4, !tbaa !10
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %5)
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %8 = load i32, ptr %7, align 4, !tbaa !10
  %9 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %8)
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load i32, ptr %10, align 4, !tbaa !10
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %11)
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %14 = load i32, ptr %13, align 4, !tbaa !10
  %15 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %14)
  %16 = load i32, ptr %1, align 4, !tbaa !10
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %16)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #9
  store i1 true, ptr @_ZL11ShouldThrow, align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #9
  invoke void @_ZN1BC2Ev(ptr noundef nonnull align 4 dereferenceable(24) %2)
          to label %18 unwind label %34

18:                                               ; preds = %0
  %19 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.9)
  %20 = getelementptr inbounds nuw i8, ptr %2, i64 20
  %21 = load i32, ptr %20, align 4, !tbaa !10
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %21)
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 16
  %24 = load i32, ptr %23, align 4, !tbaa !10
  %25 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %24)
  %26 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %27 = load i32, ptr %26, align 4, !tbaa !10
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %27)
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %30 = load i32, ptr %29, align 4, !tbaa !10
  %31 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %30)
  %32 = load i32, ptr %2, align 4, !tbaa !10
  %33 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %32)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #9
  br label %39

34:                                               ; preds = %0
  %35 = landingpad { ptr, i32 }
          catch ptr null
  %36 = extractvalue { ptr, i32 } %35, 0
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #9
  %37 = tail call ptr @__cxa_begin_catch(ptr %36) #9
  %38 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.8)
  tail call void @__cxa_end_catch()
  br label %39

39:                                               ; preds = %34, %18
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #8

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold noreturn }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn }
attributes #5 = { nofree nosync nounwind memory(none) }
attributes #6 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #7 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nofree nounwind }
attributes #9 = { nounwind }
attributes #10 = { noreturn }
attributes #11 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
!10 = !{!11, !7, i64 0}
!11 = !{!"_ZTS1A", !7, i64 0}
!12 = !{!13, !7, i64 12}
!13 = !{!"_ZTS1B", !11, i64 0, !11, i64 4, !11, i64 8, !7, i64 12, !11, i64 16, !11, i64 20}
