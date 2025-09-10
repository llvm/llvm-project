; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/inlined_cleanup.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/inlined_cleanup.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%class.Cleanup = type { [10 x i8] }

$_ZN7CleanupD2Ev = comdat any

$_ZTI7Cleanup = comdat any

$_ZTS7Cleanup = comdat any

$_ZTIP7Cleanup = comdat any

$_ZTSP7Cleanup = comdat any

@_ZTIi = external constant ptr
@.str = private unnamed_addr constant [12 x i8] c"Caught %d!\0A\00", align 1
@_ZTI7Cleanup = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS7Cleanup }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS7Cleanup = linkonce_odr dso_local constant [9 x i8] c"7Cleanup\00", comdat, align 1
@_ZTIP7Cleanup = linkonce_odr dso_local constant { ptr, ptr, i32, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv119__pointer_type_infoE, i64 2), ptr @_ZTSP7Cleanup, i32 0, ptr @_ZTI7Cleanup }, comdat, align 8
@_ZTVN10__cxxabiv119__pointer_type_infoE = external global [0 x ptr]
@_ZTSP7Cleanup = linkonce_odr dso_local constant [10 x i8] c"P7Cleanup\00", comdat, align 1
@.str.4 = private unnamed_addr constant [3 x i8] c"ap\00", align 1
@.str.5 = private unnamed_addr constant [3 x i8] c"cp\00", align 1
@.str.7 = private unnamed_addr constant [17 x i8] c"Cleanup for %s!\0A\00", align 1
@str.8 = private unnamed_addr constant [16 x i8] c"Caught cleanup!\00", align 4

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = alloca %class.Cleanup, align 4
  %2 = alloca %class.Cleanup, align 1
  invoke fastcc void @_ZL3foov()
          to label %14 unwind label %3

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %5 = extractvalue { ptr, i32 } %4, 1
  %6 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #11
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %8, label %46

8:                                                ; preds = %3
  %9 = extractvalue { ptr, i32 } %4, 0
  %10 = tail call ptr @__cxa_begin_catch(ptr %9) #11
  %11 = load i32, ptr %10, align 4, !tbaa !6
  %12 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %11)
  tail call void @__cxa_end_catch() #11
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #11
  store i16 97, ptr %1, align 4
  %13 = tail call ptr @__cxa_allocate_exception(i64 10) #11
  store i16 99, ptr %13, align 1
  invoke void @__cxa_throw(ptr nonnull %13, ptr nonnull @_ZTI7Cleanup, ptr nonnull @_ZN7CleanupD2Ev) #12
          to label %48 unwind label %15

14:                                               ; preds = %0
  unreachable

15:                                               ; preds = %8
  %16 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTI7Cleanup
  %17 = extractvalue { ptr, i32 } %16, 1
  %18 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef nonnull align 1 dereferenceable(10) %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  %19 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI7Cleanup) #11
  %20 = icmp eq i32 %17, %19
  br i1 %20, label %21, label %46

21:                                               ; preds = %15
  %22 = extractvalue { ptr, i32 } %16, 0
  %23 = call ptr @__cxa_begin_catch(ptr %22) #11
  %24 = call i32 @puts(ptr nonnull dereferenceable(1) @str.8)
  call void @__cxa_end_catch()
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #11
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %2, ptr noundef nonnull align 1 dereferenceable(3) @.str.4, i64 3, i1 false) #11
  %25 = call ptr @__cxa_allocate_exception(i64 8) #11
  %26 = invoke noalias noundef nonnull dereferenceable(10) ptr @_Znwm(i64 noundef 10) #13
          to label %27 unwind label %28

27:                                               ; preds = %21
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %26, ptr noundef nonnull align 1 dereferenceable(3) @.str.5, i64 3, i1 false) #11
  store ptr %26, ptr %25, align 16, !tbaa !10
  invoke void @__cxa_throw(ptr nonnull %25, ptr nonnull @_ZTIP7Cleanup, ptr null) #12
          to label %48 unwind label %30

28:                                               ; preds = %21
  %29 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIP7Cleanup
  call void @__cxa_free_exception(ptr %25) #11
  br label %32

30:                                               ; preds = %27
  %31 = landingpad { ptr, i32 }
          cleanup
          catch ptr @_ZTIP7Cleanup
  br label %32

32:                                               ; preds = %30, %28
  %33 = phi { ptr, i32 } [ %31, %30 ], [ %29, %28 ]
  %34 = extractvalue { ptr, i32 } %33, 1
  %35 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef nonnull align 1 dereferenceable(10) %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #11
  %36 = call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIP7Cleanup) #11
  %37 = icmp eq i32 %34, %36
  br i1 %37, label %38, label %46

38:                                               ; preds = %32
  %39 = extractvalue { ptr, i32 } %33, 0
  %40 = call ptr @__cxa_begin_catch(ptr %39) #11
  %41 = call i32 @puts(ptr nonnull dereferenceable(1) @str.8)
  %42 = icmp eq ptr %40, null
  br i1 %42, label %45, label %43

43:                                               ; preds = %38
  %44 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef nonnull align 1 dereferenceable(10) %40)
  call void @_ZdlPvm(ptr noundef nonnull %40, i64 noundef 10) #14
  br label %45

45:                                               ; preds = %43, %38
  call void @__cxa_end_catch() #11
  ret i32 0

46:                                               ; preds = %32, %15, %3
  %47 = phi { ptr, i32 } [ %33, %32 ], [ %16, %15 ], [ %4, %3 ]
  resume { ptr, i32 } %47

48:                                               ; preds = %27, %8
  unreachable
}

; Function Attrs: cold mustprogress norecurse noreturn uwtable
define internal fastcc void @_ZL3foov() unnamed_addr #1 personality ptr @__gxx_personality_v0 {
  %1 = alloca %class.Cleanup, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #11
  store i32 7173486, ptr %1, align 4
  %2 = tail call ptr @__cxa_allocate_exception(i64 4) #11
  store i32 3, ptr %2, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTIi, ptr null) #12
          to label %6 unwind label %3

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          cleanup
  %5 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef nonnull align 1 dereferenceable(10) %1)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #11
  resume { ptr, i32 } %4

6:                                                ; preds = %0
  unreachable
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #4

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

declare void @__cxa_free_exception(ptr) local_unnamed_addr

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN7CleanupD2Ev(ptr noundef nonnull align 1 dereferenceable(10) %0) unnamed_addr #5 comdat personality ptr @__gxx_personality_v0 {
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, ptr noundef nonnull %0)
  ret void
}

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #6

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znwm(i64 noundef) local_unnamed_addr #7

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #8

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #9

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #10

attributes #0 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold mustprogress norecurse noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nosync nounwind memory(none) }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { cold noreturn }
attributes #7 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nofree nounwind }
attributes #10 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #11 = { nounwind }
attributes #12 = { noreturn }
attributes #13 = { builtin allocsize(0) }
attributes #14 = { builtin nounwind }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 _ZTS7Cleanup", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
