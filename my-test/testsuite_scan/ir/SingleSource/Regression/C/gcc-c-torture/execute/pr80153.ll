; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr80153.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr80153.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buf = internal unnamed_addr global ptr null, align 8
@i = internal unnamed_addr global i32 0, align 4
@l = internal unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [7 x i8] c"oops!\0A\00", align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @check(i32 %0, i32 %1, i32 noundef %2) local_unnamed_addr #0 {
  %4 = icmp eq i32 %2, 0
  br i1 %4, label %5, label %6

5:                                                ; preds = %3
  tail call void @abort() #6
  unreachable

6:                                                ; preds = %3
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: mustprogress nofree noinline norecurse nounwind willreturn memory(write, argmem: read, inaccessiblemem: none) uwtable
define dso_local void @_fputs(ptr noundef %0) local_unnamed_addr #2 {
  store ptr %0, ptr @buf, align 8, !tbaa !6
  store i32 0, ptr @i, align 4, !tbaa !11
  %2 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %0) #7
  %3 = trunc i64 %2 to i32
  store i32 %3, ptr @l, align 4, !tbaa !11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #3

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define dso_local i8 @_fgetc() local_unnamed_addr #4 {
  %1 = load ptr, ptr @buf, align 8, !tbaa !6
  %2 = load i32, ptr @i, align 4, !tbaa !11
  %3 = sext i32 %2 to i64
  %4 = getelementptr inbounds i8, ptr %1, i64 %3
  %5 = load i8, ptr %4, align 1, !tbaa !13
  %6 = add nsw i32 %2, 1
  store i32 %6, ptr @i, align 4, !tbaa !11
  %7 = load i32, ptr @l, align 4, !tbaa !11
  %8 = icmp slt i32 %2, %7
  %9 = select i1 %8, i8 %5, i8 -1
  ret i8 %9
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  tail call void @_fputs(ptr noundef nonnull @.str)
  %1 = tail call i8 @_fgetc()
  %2 = icmp eq i8 %1, 111
  %3 = zext i1 %2 to i32
  tail call void @check(i32 poison, i32 poison, i32 noundef %3)
  %4 = tail call i8 @_fgetc()
  %5 = icmp eq i8 %4, 111
  %6 = zext i1 %5 to i32
  tail call void @check(i32 poison, i32 poison, i32 noundef %6)
  %7 = tail call i8 @_fgetc()
  %8 = icmp eq i8 %7, 112
  %9 = zext i1 %8 to i32
  tail call void @check(i32 poison, i32 poison, i32 noundef %9)
  %10 = tail call i8 @_fgetc()
  %11 = icmp eq i8 %10, 115
  %12 = zext i1 %11 to i32
  tail call void @check(i32 poison, i32 poison, i32 noundef %12)
  %13 = tail call i8 @_fgetc()
  %14 = icmp eq i8 %13, 33
  %15 = zext i1 %14 to i32
  tail call void @check(i32 poison, i32 poison, i32 noundef %15)
  %16 = tail call i8 @_fgetc()
  %17 = icmp eq i8 %16, 10
  %18 = zext i1 %17 to i32
  tail call void @check(i32 poison, i32 poison, i32 noundef %18)
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nofree noinline norecurse nounwind willreturn memory(write, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 omnipotent char", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"int", !9, i64 0}
!13 = !{!9, !9, i64 0}
