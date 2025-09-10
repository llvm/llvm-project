; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/ctor_dtor_count-2.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/ctor_dtor_count-2.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

$_ZN1AC2Ev = comdat any

$_ZN1AD2Ev = comdat any

$_ZTI1A = comdat any

$_ZTS1A = comdat any

@c = dso_local local_unnamed_addr global i32 0, align 4
@d = dso_local local_unnamed_addr global i32 0, align 4
@_ZTI1A = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, comdat, align 8
@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
@_ZTS1A = linkonce_odr dso_local constant [3 x i8] c"1A\00", comdat, align 1
@.str.2 = private unnamed_addr constant [18 x i8] c"c == %d, d == %d\0A\00", align 1
@.str.3 = private unnamed_addr constant [8 x i8] c"A() %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [9 x i8] c"~A() %d\0A\00", align 1
@.str.5 = private unnamed_addr constant [16 x i8] c"A(const A&) %d\0A\00", align 1
@str = private unnamed_addr constant [14 x i8] c"Throwing 1...\00", align 4
@str.6 = private unnamed_addr constant [8 x i8] c"Caught.\00", align 4

; Function Attrs: mustprogress noreturn uwtable
define dso_local void @_Z1fv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %2 = tail call ptr @__cxa_allocate_exception(i64 4) #8
  invoke void @_ZN1AC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %2)
          to label %3 unwind label %4

3:                                                ; preds = %0
  tail call void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTI1A, ptr nonnull @_ZN1AD2Ev) #9
  unreachable

4:                                                ; preds = %0
  %5 = landingpad { ptr, i32 }
          cleanup
  tail call void @__cxa_free_exception(ptr nonnull %2) #8
  resume { ptr, i32 } %5
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN1AC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) unnamed_addr #2 comdat {
  %2 = load i32, ptr @c, align 4, !tbaa !6
  %3 = add nsw i32 %2, 1
  store i32 %3, ptr @c, align 4, !tbaa !6
  store i32 %3, ptr %0, align 4, !tbaa !10
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %3)
  ret void
}

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_free_exception(ptr) local_unnamed_addr

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN1AD2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) unnamed_addr #3 comdat personality ptr @__gxx_personality_v0 {
  %2 = load i32, ptr %0, align 4, !tbaa !10
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %2)
  %4 = load i32, ptr @d, align 4, !tbaa !6
  %5 = add nsw i32 %4, 1
  store i32 %5, ptr @d, align 4, !tbaa !6
  ret void
}

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #4

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef range(i32 0, 2) i32 @main() local_unnamed_addr #5 personality ptr @__gxx_personality_v0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %2 = tail call ptr @__cxa_allocate_exception(i64 4) #8
  %3 = load i32, ptr @c, align 4, !tbaa !6
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @c, align 4, !tbaa !6
  store i32 %4, ptr %2, align 4, !tbaa !10
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %4)
  invoke void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTI1A, ptr nonnull @_ZN1AD2Ev) #9
          to label %6 unwind label %7

6:                                                ; preds = %0
  unreachable

7:                                                ; preds = %0
  %8 = landingpad { ptr, i32 }
          catch ptr @_ZTI1A
  %9 = extractvalue { ptr, i32 } %8, 1
  %10 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI1A) #8
  %11 = icmp eq i32 %9, %10
  br i1 %11, label %12, label %30

12:                                               ; preds = %7
  %13 = extractvalue { ptr, i32 } %8, 0
  %14 = tail call ptr @__cxa_get_exception_ptr(ptr %13) #8
  %15 = load i32, ptr @c, align 4, !tbaa !6
  %16 = add nsw i32 %15, 1
  store i32 %16, ptr @c, align 4, !tbaa !6
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %16)
  %18 = tail call ptr @__cxa_begin_catch(ptr %13) #8
  %19 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.6)
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %16)
  %21 = load i32, ptr @d, align 4, !tbaa !6
  %22 = add nsw i32 %21, 1
  store i32 %22, ptr @d, align 4, !tbaa !6
  tail call void @__cxa_end_catch()
  %23 = load i32, ptr @c, align 4, !tbaa !6
  %24 = load i32, ptr @d, align 4, !tbaa !6
  %25 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %23, i32 noundef %24)
  %26 = load i32, ptr @c, align 4, !tbaa !6
  %27 = load i32, ptr @d, align 4, !tbaa !6
  %28 = icmp ne i32 %26, %27
  %29 = zext i1 %28 to i32
  ret i32 %29

30:                                               ; preds = %7
  resume { ptr, i32 } %8
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #6

declare ptr @__cxa_get_exception_ptr(ptr) local_unnamed_addr

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #7

attributes #0 = { mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold noreturn }
attributes #5 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nosync nounwind memory(none) }
attributes #7 = { nofree nounwind }
attributes #8 = { nounwind }
attributes #9 = { noreturn }

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
