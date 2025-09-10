; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65648.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr65648.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@a = dso_local local_unnamed_addr global i32 0, align 4
@b = dso_local local_unnamed_addr global ptr null, align 8
@c = dso_local local_unnamed_addr global i32 0, align 4
@e = dso_local local_unnamed_addr global i16 1, align 4
@f = internal global i64 0, align 8
@i = dso_local local_unnamed_addr global ptr @f, align 8
@j = dso_local local_unnamed_addr global i8 0, align 4
@.str = private unnamed_addr constant [1 x i8] zeroinitializer, align 1

; Function Attrs: noinline nounwind uwtable
define dso_local void @foo(i32 noundef %0, ptr noundef %1) local_unnamed_addr #0 {
  tail call void asm sideeffect "", "r,r,~{memory}"(i32 %0, ptr %1) #4, !srcloc !6
  ret void
}

; Function Attrs: noinline nounwind uwtable
define dso_local void @bar(ptr noundef %0, i64 noundef %1) local_unnamed_addr #0 {
  %3 = alloca i64, align 8
  store i64 %1, ptr %3, align 8, !tbaa !7
  call void asm sideeffect "", "r,r,~{memory}"(ptr %0, ptr nonnull %3) #4, !srcloc !11
  %4 = load i64, ptr %3, align 8, !tbaa !7
  %5 = icmp eq i64 %4, 0
  br i1 %5, label %7, label %6

6:                                                ; preds = %2
  call void @abort() #5
  unreachable

7:                                                ; preds = %2
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  store i32 0, ptr %1, align 4, !tbaa !12
  store ptr %1, ptr @b, align 8, !tbaa !14
  %2 = load i32, ptr @a, align 4, !tbaa !12
  %3 = icmp eq i32 %2, 0
  %4 = zext i1 %3 to i8
  %5 = load i32, ptr @c, align 4, !tbaa !12
  %6 = load i16, ptr @e, align 4, !tbaa !17
  %7 = sext i16 %6 to i32
  %8 = icmp sle i32 %5, %7
  %9 = sext i1 %8 to i8
  %10 = add nsw i8 %9, %4
  store i8 %10, ptr @j, align 4, !tbaa !19
  %11 = zext i8 %10 to i64
  %12 = load ptr, ptr @i, align 8, !tbaa !20
  store i64 %11, ptr %12, align 8, !tbaa !7
  call void @foo(i32 noundef %2, ptr noundef nonnull %1)
  %13 = load i64, ptr @f, align 8, !tbaa !7
  call void @bar(ptr noundef nonnull @.str, i64 noundef %13)
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #3

attributes #0 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{i64 225}
!7 = !{!8, !8, i64 0}
!8 = !{!"long long", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i64 356}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !9, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 int", !16, i64 0}
!16 = !{!"any pointer", !9, i64 0}
!17 = !{!18, !18, i64 0}
!18 = !{!"short", !9, i64 0}
!19 = !{!9, !9, i64 0}
!20 = !{!21, !21, i64 0}
!21 = !{!"p1 long long", !16, i64 0}
