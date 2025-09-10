; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/ccc.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/SignlessTypes/ccc.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@Y = dso_local local_unnamed_addr global i32 0, align 4
@.str = private unnamed_addr constant [35 x i8] c"Program requires integer parameter\00", align 1
@F1 = dso_local local_unnamed_addr global i32 0, align 4
@F2 = dso_local local_unnamed_addr global i32 0, align 4
@F3 = dso_local local_unnamed_addr global i32 0, align 4
@F4 = dso_local local_unnamed_addr global i32 0, align 4
@.str.1 = private unnamed_addr constant [12 x i8] c"%d %hd %hu\0A\00", align 1

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 -2147483646, -2147483648) i32 @ubyteArg(i8 noundef %0) #0 {
  %2 = zext i8 %0 to i32
  %3 = add nuw nsw i32 %2, 2
  %4 = load i32, ptr @Y, align 4, !tbaa !6
  %5 = add nsw i32 %3, %4
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local i32 @sbyteArg(i8 noundef %0) #0 {
  %2 = sext i8 %0 to i32
  %3 = add nsw i32 %2, 3
  %4 = load i32, ptr @Y, align 4, !tbaa !6
  %5 = add nsw i32 %3, %4
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local range(i32 -2147483644, -2147483648) i32 @ushortArg(i16 noundef %0) #0 {
  %2 = zext i16 %0 to i32
  %3 = add nuw nsw i32 %2, 4
  %4 = load i32, ptr @Y, align 4, !tbaa !6
  %5 = add nsw i32 %3, %4
  ret i32 %5
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable
define dso_local i32 @sshortArg(i16 noundef %0) #0 {
  %2 = sext i16 %0 to i32
  %3 = add nsw i32 %2, 5
  %4 = load i32, ptr @Y, align 4, !tbaa !6
  %5 = add nsw i32 %3, %4
  ret i32 %5
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #1 {
  %3 = icmp slt i32 %0, 2
  br i1 %3, label %4, label %6

4:                                                ; preds = %2
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str)
  tail call void @exit(i32 noundef 1) #5
  unreachable

6:                                                ; preds = %2
  store i32 ptrtoint (ptr @ubyteArg to i32), ptr @F1, align 4, !tbaa !6
  store i32 ptrtoint (ptr @sbyteArg to i32), ptr @F2, align 4, !tbaa !6
  store i32 ptrtoint (ptr @ushortArg to i32), ptr @F3, align 4, !tbaa !6
  store i32 ptrtoint (ptr @sshortArg to i32), ptr @F4, align 4, !tbaa !6
  %7 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %8 = load ptr, ptr %7, align 8, !tbaa !10
  %9 = tail call i64 @strtol(ptr noundef nonnull captures(none) %8, ptr noundef null, i32 noundef 10) #6
  %10 = trunc i64 %9 to i32
  %11 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %10, i32 noundef -1, i32 noundef 65535)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #4

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { cold noreturn nounwind }
attributes #6 = { nounwind }

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
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"p1 omnipotent char", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
