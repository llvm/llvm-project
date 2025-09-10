; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/exception.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/exception.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@_ZTIi = external constant ptr
@str = private unnamed_addr constant [13 x i8] c"int31 branch\00", align 4

; Function Attrs: mustprogress uwtable
define dso_local noundef double @_Z10throw_testdi(double noundef returned %0, i32 noundef %1) local_unnamed_addr #0 {
  switch i32 %1, label %11 [
    i32 0, label %3
    i32 1, label %5
    i32 2, label %7
    i32 3, label %9
  ]

3:                                                ; preds = %2
  %4 = tail call ptr @__cxa_allocate_exception(i64 4) #5
  store i32 10, ptr %4, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %4, ptr nonnull @_ZTIi, ptr null) #6
  unreachable

5:                                                ; preds = %2
  %6 = tail call ptr @__cxa_allocate_exception(i64 4) #5
  store i32 1, ptr %6, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %6, ptr nonnull @_ZTIi, ptr null) #6
  unreachable

7:                                                ; preds = %2
  %8 = tail call ptr @__cxa_allocate_exception(i64 4) #5
  store i32 2, ptr %8, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %8, ptr nonnull @_ZTIi, ptr null) #6
  unreachable

9:                                                ; preds = %2
  %10 = tail call ptr @__cxa_allocate_exception(i64 4) #5
  store i32 3, ptr %10, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %10, ptr nonnull @_ZTIi, ptr null) #6
  unreachable

11:                                               ; preds = %2
  ret double %0
}

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #1

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 personality ptr @__gxx_personality_v0 {
  %1 = tail call ptr @__cxa_allocate_exception(i64 4) #5
  store i32 3, ptr %1, align 16, !tbaa !6
  invoke void @__cxa_throw(ptr nonnull %1, ptr nonnull @_ZTIi, ptr null) #6
          to label %2 unwind label %3

2:                                                ; preds = %0
  unreachable

3:                                                ; preds = %0
  %4 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %5 = extractvalue { ptr, i32 } %4, 1
  %6 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #5
  %7 = icmp eq i32 %5, %6
  br i1 %7, label %8, label %12

8:                                                ; preds = %3
  %9 = extractvalue { ptr, i32 } %4, 0
  %10 = tail call ptr @__cxa_begin_catch(ptr %9) #5
  %11 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @__cxa_end_catch() #5
  ret i32 0

12:                                               ; preds = %3
  resume { ptr, i32 } %4
}

declare i32 @__gxx_personality_v0(...)

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #3

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #4

attributes #0 = { mustprogress uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold noreturn }
attributes #2 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nosync nounwind memory(none) }
attributes #4 = { nofree nounwind }
attributes #5 = { nounwind }
attributes #6 = { noreturn }

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
