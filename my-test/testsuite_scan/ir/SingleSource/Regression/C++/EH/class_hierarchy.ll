; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/class_hierarchy.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/class_hierarchy.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

$_ZN4BaseC2Ej = comdat any

$_ZN7DerivedC2Ej = comdat any

$_ZN6UnusedC2Ev = comdat any

$_ZN4BaseD0Ev = comdat any

$_ZN7DerivedD0Ev = comdat any

$_ZN6UnusedD0Ev = comdat any

$_ZN7Unused2D0Ev = comdat any

$_ZTI4Base = comdat any

$_ZTS4Base = comdat any

$_ZTI7Derived = comdat any

$_ZTS7Derived = comdat any

$_ZTI6Unused = comdat any

$_ZTS6Unused = comdat any

$_ZTI7Unused2 = comdat any

$_ZTS7Unused2 = comdat any

$_ZTV4Base = comdat any

$_ZTV7Derived = comdat any

$_ZTV6Unused = comdat any

$_ZTV7Unused2 = comdat any

@_ZTI4Base = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS4Base, ptr @_ZTISt9exception }, comdat, align 8
@_ZTVN10__cxxabiv120__si_class_type_infoE = external global [0 x ptr]
@_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", comdat, align 1
@_ZTISt9exception = external constant ptr
@_ZTI7Derived = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS7Derived, ptr @_ZTI4Base }, comdat, align 8
@_ZTS7Derived = linkonce_odr dso_local constant [9 x i8] c"7Derived\00", comdat, align 1
@_ZTI6Unused = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS6Unused, ptr @_ZTI4Base }, comdat, align 8
@_ZTS6Unused = linkonce_odr dso_local constant [8 x i8] c"6Unused\00", comdat, align 1
@_ZTI7Unused2 = linkonce_odr dso_local constant { ptr, ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv120__si_class_type_infoE, i64 2), ptr @_ZTS7Unused2, ptr @_ZTISt9exception }, comdat, align 8
@_ZTS7Unused2 = linkonce_odr dso_local constant [9 x i8] c"7Unused2\00", comdat, align 1
@.str = private unnamed_addr constant [6 x i8] c"what?\00", align 1
@_ZTIPKc = external constant ptr
@.str.2 = private unnamed_addr constant [22 x i8] c"Caught exception: %s\0A\00", align 1
@_ZTV4Base = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI4Base, ptr @_ZNSt9exceptionD2Ev, ptr @_ZN4BaseD0Ev, ptr @_ZNKSt9exception4whatEv] }, comdat, align 8
@.str.3 = private unnamed_addr constant [7 x i8] c"N < 10\00", align 1
@.str.4 = private unnamed_addr constant [107 x i8] c"/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/class_hierarchy.cpp\00", align 1
@__PRETTY_FUNCTION__._ZN4BaseC2Ej = private unnamed_addr constant [25 x i8] c"Base::Base(unsigned int)\00", align 1
@.str.5 = private unnamed_addr constant [5 x i8] c"base\00", align 1
@_ZTVSt9exception = external unnamed_addr constant { [5 x ptr] }, align 8
@.str.6 = private unnamed_addr constant [12 x i8] c"n: %s class\00", align 1
@_ZTV7Derived = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI7Derived, ptr @_ZNSt9exceptionD2Ev, ptr @_ZN7DerivedD0Ev, ptr @_ZNKSt9exception4whatEv] }, comdat, align 8
@.str.7 = private unnamed_addr constant [7 x i8] c"n < 20\00", align 1
@__PRETTY_FUNCTION__._ZN7DerivedC2Ej = private unnamed_addr constant [31 x i8] c"Derived::Derived(unsigned int)\00", align 1
@.str.8 = private unnamed_addr constant [8 x i8] c"derived\00", align 1
@_ZTV6Unused = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI6Unused, ptr @_ZNSt9exceptionD2Ev, ptr @_ZN6UnusedD0Ev, ptr @_ZNKSt9exception4whatEv] }, comdat, align 8
@_ZTV7Unused2 = linkonce_odr dso_local unnamed_addr constant { [5 x ptr] } { [5 x ptr] [ptr null, ptr @_ZTI7Unused2, ptr @_ZNSt9exceptionD2Ev, ptr @_ZN7Unused2D0Ev, ptr @_ZNKSt9exception4whatEv] }, comdat, align 8
@str = private unnamed_addr constant [25 x i8] c"Caught unknown exception\00", align 4

; Function Attrs: mustprogress noreturn uwtable
define dso_local void @_Z4funcj(i32 noundef %0) local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %2 = icmp ult i32 %0, 10
  br i1 %2, label %3, label %8

3:                                                ; preds = %1
  %4 = tail call ptr @__cxa_allocate_exception(i64 24) #12
  invoke void @_ZN4BaseC2Ej(ptr noundef nonnull align 8 dereferenceable(24) %4, i32 noundef %0)
          to label %5 unwind label %6

5:                                                ; preds = %3
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTI4Base, ptr nonnull @_ZNSt9exceptionD2Ev) #13
  unreachable

6:                                                ; preds = %3
  %7 = landingpad { ptr, i32 }
          cleanup
  br label %31

8:                                                ; preds = %1
  %9 = icmp ult i32 %0, 20
  br i1 %9, label %10, label %15

10:                                               ; preds = %8
  %11 = tail call ptr @__cxa_allocate_exception(i64 24) #12
  invoke void @_ZN7DerivedC2Ej(ptr noundef nonnull align 8 dereferenceable(24) %11, i32 noundef %0)
          to label %12 unwind label %13

12:                                               ; preds = %10
  tail call void @__cxa_throw(ptr nonnull %11, ptr nonnull @_ZTI7Derived, ptr nonnull @_ZNSt9exceptionD2Ev) #13
  unreachable

13:                                               ; preds = %10
  %14 = landingpad { ptr, i32 }
          cleanup
  br label %31

15:                                               ; preds = %8
  %16 = icmp eq i32 %0, 20
  br i1 %16, label %17, label %22

17:                                               ; preds = %15
  %18 = tail call ptr @__cxa_allocate_exception(i64 24) #12
  invoke void @_ZN6UnusedC2Ev(ptr noundef nonnull align 8 dereferenceable(24) %18)
          to label %19 unwind label %20

19:                                               ; preds = %17
  tail call void @__cxa_throw(ptr nonnull %18, ptr nonnull @_ZTI6Unused, ptr nonnull @_ZNSt9exceptionD2Ev) #13
  unreachable

20:                                               ; preds = %17
  %21 = landingpad { ptr, i32 }
          cleanup
  br label %31

22:                                               ; preds = %15
  %23 = icmp ult i32 %0, 22
  br i1 %23, label %24, label %26

24:                                               ; preds = %22
  %25 = tail call ptr @__cxa_allocate_exception(i64 8) #12
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV7Unused2, i64 16), ptr %25, align 8, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %25, ptr nonnull @_ZTI7Unused2, ptr nonnull @_ZNSt9exceptionD2Ev) #13
  unreachable

26:                                               ; preds = %22
  %27 = icmp eq i32 %0, 22
  %28 = tail call ptr @__cxa_allocate_exception(i64 8) #12
  br i1 %27, label %29, label %30

29:                                               ; preds = %26
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTVSt9exception, i64 16), ptr %28, align 8, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %28, ptr nonnull @_ZTISt9exception, ptr nonnull @_ZNSt9exceptionD1Ev) #13
  unreachable

30:                                               ; preds = %26
  store ptr @.str, ptr %28, align 16, !tbaa !9
  tail call void @__cxa_throw(ptr nonnull %28, ptr nonnull @_ZTIPKc, ptr null) #13
  unreachable

31:                                               ; preds = %20, %13, %6
  %32 = phi ptr [ %18, %20 ], [ %11, %13 ], [ %4, %6 ]
  %33 = phi { ptr, i32 } [ %21, %20 ], [ %14, %13 ], [ %7, %6 ]
  tail call void @__cxa_free_exception(ptr nonnull %32) #12
  resume { ptr, i32 } %33
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN4BaseC2Ej(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #1 comdat personality ptr @__gxx_personality_v0 {
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV4Base, i64 16), ptr %0, align 8, !tbaa !6
  %3 = icmp ult i32 %1, 10
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @__assert_fail(ptr noundef nonnull @.str.3, ptr noundef nonnull @.str.4, i32 noundef 18, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN4BaseC2Ej) #14
  unreachable

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 %1, ptr %6, align 8, !tbaa !13
  %7 = invoke noalias noundef nonnull dereferenceable(14) ptr @_Znam(i64 noundef 14) #15
          to label %8 unwind label %11

8:                                                ; preds = %5
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %7, ptr %9, align 8, !tbaa !17
  %10 = tail call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %7, ptr noundef nonnull dereferenceable(1) @.str.6, ptr noundef nonnull @.str.5) #12
  ret void

11:                                               ; preds = %5
  %12 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #12
  resume { ptr, i32 } %12
}

declare i32 @__gxx_personality_v0(...)

declare void @__cxa_free_exception(ptr) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #2

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN7DerivedC2Ej(ptr noundef nonnull align 8 dereferenceable(24) %0, i32 noundef %1) unnamed_addr #1 comdat personality ptr @__gxx_personality_v0 {
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV7Derived, i64 16), ptr %0, align 8, !tbaa !6
  %3 = icmp ult i32 %1, 20
  br i1 %3, label %5, label %4

4:                                                ; preds = %2
  tail call void @__assert_fail(ptr noundef nonnull @.str.7, ptr noundef nonnull @.str.4, i32 noundef 30, ptr noundef nonnull @__PRETTY_FUNCTION__._ZN7DerivedC2Ej) #14
  unreachable

5:                                                ; preds = %2
  %6 = add nsw i32 %1, -10
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 %6, ptr %7, align 8, !tbaa !13
  %8 = invoke noalias noundef nonnull dereferenceable(17) ptr @_Znam(i64 noundef 17) #15
          to label %9 unwind label %12

9:                                                ; preds = %5
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %8, ptr %10, align 8, !tbaa !17
  %11 = tail call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %8, ptr noundef nonnull dereferenceable(1) @.str.6, ptr noundef nonnull @.str.8) #12
  ret void

12:                                               ; preds = %5
  %13 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) #12
  resume { ptr, i32 } %13
}

; Function Attrs: mustprogress uwtable
define linkonce_odr dso_local void @_ZN6UnusedC2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #1 comdat personality ptr @__gxx_personality_v0 {
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV4Base, i64 16), ptr %0, align 8, !tbaa !6
  %2 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 0, ptr %2, align 8, !tbaa !13
  %3 = invoke noalias noundef nonnull dereferenceable(14) ptr @_Znam(i64 noundef 14) #15
          to label %6 unwind label %4

4:                                                ; preds = %1
  %5 = landingpad { ptr, i32 }
          cleanup
  tail call void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) #12
  resume { ptr, i32 } %5

6:                                                ; preds = %1
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store ptr %3, ptr %7, align 8, !tbaa !17
  %8 = tail call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) @.str.6, ptr noundef nonnull @.str.5) #12
  store ptr getelementptr inbounds nuw inrange(-16, 24) (i8, ptr @_ZTV6Unused, i64 16), ptr %0, align 8, !tbaa !6
  ret void
}

; Function Attrs: nounwind
declare void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #3

; Function Attrs: nounwind
declare void @_ZNSt9exceptionD1Ev(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #3

; Function Attrs: mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #4 personality ptr @__gxx_personality_v0 {
  br label %2

1:                                                ; preds = %21
  ret i32 0

2:                                                ; preds = %0, %21
  %3 = phi i32 [ 0, %0 ], [ %22, %21 ]
  invoke void @_Z4funcj(i32 noundef %3)
          to label %20 unwind label %4

4:                                                ; preds = %2
  %5 = landingpad { ptr, i32 }
          catch ptr @_ZTI7Derived
          catch ptr @_ZTI4Base
          catch ptr @_ZTISt9exception
          catch ptr null
  %6 = extractvalue { ptr, i32 } %5, 0
  %7 = extractvalue { ptr, i32 } %5, 1
  %8 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI7Derived) #12
  %9 = icmp eq i32 %7, %8
  br i1 %9, label %10, label %24

10:                                               ; preds = %4
  %11 = tail call ptr @__cxa_begin_catch(ptr %6) #12
  %12 = getelementptr inbounds nuw i8, ptr %11, i64 8
  %13 = load i32, ptr %12, align 8, !tbaa !13
  %14 = trunc i32 %13 to i8
  %15 = add i8 %14, 48
  %16 = getelementptr inbounds nuw i8, ptr %11, i64 16
  %17 = load ptr, ptr %16, align 8, !tbaa !17
  store i8 %15, ptr %17, align 1, !tbaa !18
  %18 = load ptr, ptr %16, align 8, !tbaa !17
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, ptr noundef %18)
  br label %21

20:                                               ; preds = %2
  unreachable

21:                                               ; preds = %47, %41, %27, %10
  tail call void @__cxa_end_catch()
  %22 = add nuw nsw i32 %3, 1
  %23 = icmp eq i32 %22, 25
  br i1 %23, label %1, label %2, !llvm.loop !19

24:                                               ; preds = %4
  %25 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTI4Base) #12
  %26 = icmp eq i32 %7, %25
  br i1 %26, label %27, label %37

27:                                               ; preds = %24
  %28 = tail call ptr @__cxa_begin_catch(ptr %6) #12
  %29 = getelementptr inbounds nuw i8, ptr %28, i64 8
  %30 = load i32, ptr %29, align 8, !tbaa !13
  %31 = trunc i32 %30 to i8
  %32 = add i8 %31, 48
  %33 = getelementptr inbounds nuw i8, ptr %28, i64 16
  %34 = load ptr, ptr %33, align 8, !tbaa !17
  store i8 %32, ptr %34, align 1, !tbaa !18
  %35 = load ptr, ptr %33, align 8, !tbaa !17
  %36 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, ptr noundef %35)
  br label %21

37:                                               ; preds = %24
  %38 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTISt9exception) #12
  %39 = icmp eq i32 %7, %38
  %40 = tail call ptr @__cxa_begin_catch(ptr %6) #12
  br i1 %39, label %41, label %47

41:                                               ; preds = %37
  %42 = load ptr, ptr %40, align 8, !tbaa !6
  %43 = getelementptr inbounds nuw i8, ptr %42, i64 16
  %44 = load ptr, ptr %43, align 8
  %45 = tail call noundef ptr %44(ptr noundef nonnull align 8 dereferenceable(8) %40) #12
  %46 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, ptr noundef %45)
  br label %21

47:                                               ; preds = %37
  %48 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  br label %21
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #5

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #6

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: cold noreturn nounwind
declare void @__assert_fail(ptr noundef, ptr noundef, i32 noundef, ptr noundef) local_unnamed_addr #7

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN4BaseD0Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #8 comdat {
  tail call void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) #12
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 24) #16
  ret void
}

; Function Attrs: nounwind
declare noundef ptr @_ZNKSt9exception4whatEv(ptr noundef nonnull align 8 dereferenceable(8)) unnamed_addr #3

; Function Attrs: nobuiltin allocsize(0)
declare noundef nonnull ptr @_Znam(i64 noundef) local_unnamed_addr #9

; Function Attrs: nofree nounwind
declare noundef i32 @sprintf(ptr noalias noundef writeonly captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #6

; Function Attrs: nobuiltin nounwind
declare void @_ZdlPvm(ptr noundef, i64 noundef) local_unnamed_addr #10

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN7DerivedD0Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #8 comdat {
  tail call void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) #12
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 24) #16
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN6UnusedD0Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) unnamed_addr #8 comdat {
  tail call void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(24) %0) #12
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 24) #16
  ret void
}

; Function Attrs: inlinehint mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN7Unused2D0Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) unnamed_addr #8 comdat {
  tail call void @_ZNSt9exceptionD2Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #12
  tail call void @_ZdlPvm(ptr noundef nonnull %0, i64 noundef 8) #16
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #11

attributes #0 = { mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold noreturn }
attributes #3 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nosync nounwind memory(none) }
attributes #6 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { cold noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { inlinehint mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nobuiltin allocsize(0) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nobuiltin nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { nofree nounwind }
attributes #12 = { nounwind }
attributes #13 = { noreturn }
attributes #14 = { cold noreturn nounwind }
attributes #15 = { builtin allocsize(0) }
attributes #16 = { builtin nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"vtable pointer", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"p1 omnipotent char", !11, i64 0}
!11 = !{!"any pointer", !12, i64 0}
!12 = !{!"omnipotent char", !8, i64 0}
!13 = !{!14, !16, i64 8}
!14 = !{!"_ZTS4Base", !15, i64 0, !16, i64 8, !10, i64 16}
!15 = !{!"_ZTSSt9exception"}
!16 = !{!"int", !12, i64 0}
!17 = !{!14, !10, i64 16}
!18 = !{!12, !12, i64 0}
!19 = distinct !{!19, !20}
!20 = !{!"llvm.loop.mustprogress"}
