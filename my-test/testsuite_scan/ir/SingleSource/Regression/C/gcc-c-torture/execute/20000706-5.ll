; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20000706-5.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20000706-5.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.baz = type { i32, i32, i32 }

@c = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: nofree nounwind uwtable
define dso_local void @bar(i32 noundef %0) local_unnamed_addr #0 {
  %2 = load ptr, ptr @c, align 8, !tbaa !6
  %3 = load i32, ptr %2, align 4, !tbaa !11
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %5, label %15

5:                                                ; preds = %1
  %6 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %7 = load i32, ptr %6, align 4, !tbaa !14
  %8 = icmp eq i32 %7, 2
  br i1 %8, label %9, label %15

9:                                                ; preds = %5
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %11 = load i32, ptr %10, align 4, !tbaa !15
  %12 = icmp ne i32 %11, 3
  %13 = icmp ne i32 %0, 4
  %14 = or i1 %13, %12
  br i1 %14, label %15, label %16

15:                                               ; preds = %9, %5, %1
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %9
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local void @foo([2 x i64] %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = alloca %struct.baz, align 8
  %4 = extractvalue [2 x i64] %0, 0
  %5 = extractvalue [2 x i64] %0, 1
  store i64 %4, ptr %3, align 8
  %6 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %7 = trunc i64 %5 to i32
  store i32 %7, ptr %6, align 8
  store ptr %3, ptr @c, align 8, !tbaa !6
  %8 = icmp eq i64 %4, 8589934593
  br i1 %8, label %9, label %13

9:                                                ; preds = %2
  %10 = icmp ne i32 %7, 3
  %11 = icmp ne i32 %1, 4
  %12 = or i1 %11, %10
  br i1 %12, label %13, label %14

13:                                               ; preds = %9, %2
  call void @abort() #5
  unreachable

14:                                               ; preds = %9
  ret void
}

; Function Attrs: nofree noreturn nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca %struct.baz, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1)
  store ptr %1, ptr @c, align 8, !tbaa !6
  call void @llvm.lifetime.end.p0(ptr nonnull %1)
  call void @exit(i32 noundef 0) #5
  unreachable
}

; Function Attrs: nofree noreturn
declare void @exit(i32 noundef) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noreturn nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS3baz", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !13, i64 0}
!12 = !{!"baz", !13, i64 0, !13, i64 4, !13, i64 8}
!13 = !{!"int", !9, i64 0}
!14 = !{!12, !13, i64 4}
!15 = !{!12, !13, i64 8}
