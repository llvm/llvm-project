; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SetjmpLongjmp/C++/C++Catch.cpp'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SetjmpLongjmp/C++/C++Catch.cpp"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__jmp_buf_tag = type { [22 x i64], i32, %struct.__sigset_t }
%struct.__sigset_t = type { [16 x i64] }

@buf = dso_local global [1 x %struct.__jmp_buf_tag] zeroinitializer, align 8
@.str.2 = private unnamed_addr constant [5 x i8] c"main\00", align 1
@.str.5 = private unnamed_addr constant [29 x i8] c"Destroying C in function %s\0A\00", align 1
@str = private unnamed_addr constant [31 x i8] c"Longjmping from foo() function\00", align 4
@str.7 = private unnamed_addr constant [45 x i8] c"longjmp throw NOT caught in catch(...) block\00", align 4

; Function Attrs: mustprogress noreturn nounwind uwtable
define dso_local void @_Z3foov() local_unnamed_addr #0 personality ptr @__gxx_personality_v0 {
  %1 = tail call i32 @puts(ptr nonnull dereferenceable(1) @str)
  tail call void @longjmp(ptr noundef nonnull @buf, i32 noundef 37) #6
  unreachable
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

declare i32 @__gxx_personality_v0(...)

; Function Attrs: noreturn nounwind
declare void @longjmp(ptr noundef, i32 noundef) local_unnamed_addr #2

; Function Attrs: mustprogress norecurse nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #3 personality ptr @__gxx_personality_v0 {
  %1 = call i32 @_setjmp(ptr noundef nonnull @buf) #7
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %5

3:                                                ; preds = %0
  %4 = call i32 @puts(ptr nonnull dereferenceable(1) @str)
  call void @longjmp(ptr noundef nonnull @buf, i32 noundef 37) #6
  unreachable

5:                                                ; preds = %0
  %6 = call i32 @puts(ptr nonnull dereferenceable(1) @str.7)
  %7 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, ptr noundef nonnull @.str.2)
  ret i32 0
}

; Function Attrs: nounwind returns_twice
declare i32 @_setjmp(ptr noundef) local_unnamed_addr #4

; Function Attrs: nofree nounwind
declare noundef i32 @puts(ptr noundef readonly captures(none)) local_unnamed_addr #5

attributes #0 = { mustprogress noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind returns_twice "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind returns_twice }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
