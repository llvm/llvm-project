; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr79327.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr79327.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [6 x i8] c"%#hho\00", align 1
@a = dso_local global i32 0, align 4
@.str.1 = private unnamed_addr constant [6 x i8] c"%#hhx\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = alloca [64 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  %2 = load volatile i32, ptr @a, align 4, !tbaa !6
  %3 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %2) #4
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %6, label %5

5:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

6:                                                ; preds = %0
  %7 = load volatile i32, ptr @a, align 4, !tbaa !6
  %8 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %7) #4
  %9 = icmp eq i32 %8, 1
  br i1 %9, label %11, label %10

10:                                               ; preds = %6
  tail call void @abort() #5
  unreachable

11:                                               ; preds = %6
  store volatile i32 1, ptr @a, align 4, !tbaa !6
  %12 = load volatile i32, ptr @a, align 4, !tbaa !6
  %13 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %12) #4
  %14 = icmp eq i32 %13, 2
  br i1 %14, label %16, label %15

15:                                               ; preds = %11
  tail call void @abort() #5
  unreachable

16:                                               ; preds = %11
  %17 = load volatile i32, ptr @a, align 4, !tbaa !6
  %18 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %17) #4
  %19 = icmp eq i32 %18, 3
  br i1 %19, label %21, label %20

20:                                               ; preds = %16
  tail call void @abort() #5
  unreachable

21:                                               ; preds = %16
  store volatile i32 127, ptr @a, align 4, !tbaa !6
  %22 = load volatile i32, ptr @a, align 4, !tbaa !6
  %23 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %22) #4
  %24 = icmp eq i32 %23, 4
  br i1 %24, label %26, label %25

25:                                               ; preds = %21
  tail call void @abort() #5
  unreachable

26:                                               ; preds = %21
  %27 = load volatile i32, ptr @a, align 4, !tbaa !6
  %28 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %1, ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %27) #4
  %29 = icmp eq i32 %28, 4
  br i1 %29, label %31, label %30

30:                                               ; preds = %26
  tail call void @abort() #5
  unreachable

31:                                               ; preds = %26
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind
declare noundef i32 @sprintf(ptr noalias noundef writeonly captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
