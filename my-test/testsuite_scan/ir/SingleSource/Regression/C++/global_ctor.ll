; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/global_ctor.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/global_ctor.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.foo = type { i32 }
%struct.bar = type { i8 }

$_ZN3fooD2Ev = comdat any

$_ZN3barD2Ev = comdat any

@CN = dso_local local_unnamed_addr global i32 0, align 4
@DN = dso_local local_unnamed_addr global i32 0, align 4
@Constructor1 = dso_local global %struct.foo zeroinitializer, align 4
@__dso_handle = external hidden global i8
@Constructor2 = dso_local global %struct.foo zeroinitializer, align 4
@Destructor1 = dso_local global %struct.bar zeroinitializer, align 1
@.str.3 = private unnamed_addr constant [16 x i8] c"Foo ctor %d %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [16 x i8] c"Foo dtor %d %d\0A\00", align 1
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_global_ctor.cpp, ptr null }]
@str = private unnamed_addr constant [9 x i8] c"bar dtor\00", align 4
@str.6 = private unnamed_addr constant [5 x i8] c"main\00", align 4

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN3fooD2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %2 = load i32, ptr %0, align 4, !tbaa !6
  %3 = load i32, ptr @DN, align 4, !tbaa !11
  %4 = add nsw i32 %3, 1
  store i32 %4, ptr @DN, align 4, !tbaa !11
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %2, i32 noundef %3)
  ret void
}

; Function Attrs: nofree nounwind
declare i32 @__cxa_atexit(ptr, ptr, ptr) local_unnamed_addr #1

; Function Attrs: mustprogress nounwind uwtable
define linkonce_odr dso_local void @_ZN3barD2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0) unnamed_addr #0 comdat personality ptr @__gxx_personality_v0 {
  %2 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  ret void
}

; Function Attrs: mustprogress nofree norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.6)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nofree nounwind uwtable
define internal void @_GLOBAL__sub_I_global_ctor.cpp() #4 section ".text.startup" {
  store i32 7, ptr @Constructor1, align 4, !tbaa !6
  %1 = load i32, ptr @CN, align 4, !tbaa !11
  %2 = add nsw i32 %1, 1
  store i32 %2, ptr @CN, align 4, !tbaa !11
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef 7, i32 noundef %1)
  %4 = tail call i32 @__cxa_atexit(ptr nonnull @_ZN3fooD2Ev, ptr nonnull @Constructor1, ptr nonnull @__dso_handle) #5
  store i32 12, ptr @Constructor2, align 4, !tbaa !6
  %5 = load i32, ptr @CN, align 4, !tbaa !11
  %6 = add nsw i32 %5, 1
  store i32 %6, ptr @CN, align 4, !tbaa !11
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef 12, i32 noundef %5)
  %8 = tail call i32 @__cxa_atexit(ptr nonnull @_ZN3fooD2Ev, ptr nonnull @Constructor2, ptr nonnull @__dso_handle) #5
  %9 = tail call i32 @__cxa_atexit(ptr nonnull @_ZN3barD2Ev, ptr nonnull @Destructor1, ptr nonnull @__dso_handle) #5
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #1

attributes #0 = { mustprogress nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind }
attributes #2 = { mustprogress nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"_ZTS3foo", !8, i64 0}
!8 = !{!"int", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C++ TBAA"}
!11 = !{!8, !8, i64 0}
