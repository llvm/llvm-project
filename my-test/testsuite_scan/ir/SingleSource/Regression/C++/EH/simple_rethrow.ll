; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/simple_rethrow.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C++/EH/simple_rethrow.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

$__clang_call_terminate = comdat any

@_ZTIi = external constant ptr
@.str.2 = private unnamed_addr constant [16 x i8] c"Caught int: %d\0A\00", align 1
@str = private unnamed_addr constant [13 x i8] c"Throwing int\00", align 4
@str.3 = private unnamed_addr constant [32 x i8] c"Caught something, rethrowing...\00", align 4

; Function Attrs: cold mustprogress noreturn uwtable
define dso_local void @_Z6throwsv() local_unnamed_addr #0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  %2 = tail call ptr @__cxa_allocate_exception(i64 4) #8
  store i32 16, ptr %2, align 16, !tbaa !6
  tail call void @__cxa_throw(ptr nonnull %2, ptr nonnull @_ZTIi, ptr null) #9
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

declare ptr @__cxa_allocate_exception(i64) local_unnamed_addr

; Function Attrs: cold noreturn
declare void @__cxa_throw(ptr, ptr, ptr) local_unnamed_addr #2

; Function Attrs: cold mustprogress noreturn uwtable
define dso_local void @_Z11callsthrowsv() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  invoke void @_Z6throwsv()
          to label %8 unwind label %1

1:                                                ; preds = %0
  %2 = landingpad { ptr, i32 }
          catch ptr null
  %3 = extractvalue { ptr, i32 } %2, 0
  %4 = tail call ptr @__cxa_begin_catch(ptr %3) #8
  %5 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str.3)
  invoke void @__cxa_rethrow() #9
          to label %13 unwind label %6

6:                                                ; preds = %1
  %7 = landingpad { ptr, i32 }
          cleanup
  invoke void @__cxa_end_catch()
          to label %9 unwind label %10

8:                                                ; preds = %0
  unreachable

9:                                                ; preds = %6
  resume { ptr, i32 } %7

10:                                               ; preds = %6
  %11 = landingpad { ptr, i32 }
          catch ptr null
  %12 = extractvalue { ptr, i32 } %11, 0
  tail call void @__clang_call_terminate(ptr %12) #10
  unreachable

13:                                               ; preds = %1
  unreachable
}

declare i32 @__gxx_personality_v0(...)

declare ptr @__cxa_begin_catch(ptr) local_unnamed_addr

declare void @__cxa_rethrow() local_unnamed_addr

declare void @__cxa_end_catch() local_unnamed_addr

; Function Attrs: noinline noreturn nounwind uwtable
define linkonce_odr hidden void @__clang_call_terminate(ptr noundef %0) local_unnamed_addr #3 comdat {
  %2 = tail call ptr @__cxa_begin_catch(ptr %0) #8
  tail call void @_ZSt9terminatev() #10
  unreachable
}

; Function Attrs: cold nofree noreturn
declare void @_ZSt9terminatev() local_unnamed_addr #4

; Function Attrs: cold mustprogress norecurse uwtable
define dso_local noundef range(i32 -2147483648, 2147483632) i32 @main() local_unnamed_addr #5 personality ptr @__gxx_personality_v0 {
  invoke void @_Z11callsthrowsv()
          to label %12 unwind label %1

1:                                                ; preds = %0
  %2 = landingpad { ptr, i32 }
          catch ptr @_ZTIi
  %3 = extractvalue { ptr, i32 } %2, 1
  %4 = tail call i32 @llvm.eh.typeid.for.p0(ptr nonnull @_ZTIi) #8
  %5 = icmp eq i32 %3, %4
  br i1 %5, label %6, label %13

6:                                                ; preds = %1
  %7 = extractvalue { ptr, i32 } %2, 0
  %8 = tail call ptr @__cxa_begin_catch(ptr %7) #8
  %9 = load i32, ptr %8, align 4, !tbaa !6
  %10 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %9)
  %11 = add nsw i32 %9, -16
  tail call void @__cxa_end_catch() #8
  ret i32 %11

12:                                               ; preds = %0
  unreachable

13:                                               ; preds = %1
  resume { ptr, i32 } %2
}

; Function Attrs: nofree nosync nounwind memory(none)
declare i32 @llvm.eh.typeid.for.p0(ptr) #6

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #7

attributes #0 = { cold mustprogress noreturn uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold noreturn }
attributes #3 = { noinline noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { cold nofree noreturn }
attributes #5 = { cold mustprogress norecurse uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nosync nounwind memory(none) }
attributes #7 = { nofree nounwind }
attributes #8 = { nounwind }
attributes #9 = { noreturn }
attributes #10 = { noreturn nounwind }

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
