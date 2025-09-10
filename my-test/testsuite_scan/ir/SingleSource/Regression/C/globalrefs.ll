; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/globalrefs.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/globalrefs.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.test = type { i32, %struct.anon, i32, ptr }
%struct.anon = type { i32, i32 }

@TestArray = dso_local global [10 x %struct.test] zeroinitializer, align 8
@TestArrayPtr = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @TestArray, i64 72), align 8
@Test1 = dso_local global %struct.test zeroinitializer, align 8
@Aptr = dso_local local_unnamed_addr global ptr @Test1, align 8
@Yptr = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @Test1, i64 8), align 8
@NextPtr = dso_local local_unnamed_addr global ptr getelementptr inbounds nuw (i8, ptr @Test1, i64 16), align 8
@.str = private unnamed_addr constant [4 x i8] c" %d\00", align 1
@.str.2 = private unnamed_addr constant [35 x i8] c"&TestArray[3] - TestArray = 0x%lx\0A\00", align 1
@.str.3 = private unnamed_addr constant [30 x i8] c"Xptr - Aptr          = 0x%lx\0A\00", align 1
@.str.4 = private unnamed_addr constant [31 x i8] c"NextPtr - Xptr       = 0x%lx\0A\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local void @printdiff(ptr noundef %0, ptr noundef %1) local_unnamed_addr #0 {
  %3 = ptrtoint ptr %0 to i64
  %4 = ptrtoint ptr %1 to i64
  %5 = sub i64 %3, %4
  %6 = trunc i64 %5 to i32
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %6)
  ret void
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: nofree nounwind uwtable
define dso_local range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 8)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 8)
  %5 = tail call i32 @putchar(i32 10)
  %6 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i64 noundef 3)
  %7 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef 8)
  %8 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i64 noundef 8)
  %9 = load ptr, ptr @TestArrayPtr, align 8, !tbaa !6
  %10 = ptrtoint ptr %9 to i64
  %11 = sub i64 %10, ptrtoint (ptr @TestArray to i64)
  %12 = udiv i64 %11, 24
  %13 = mul nuw i64 %12, 24
  %14 = icmp eq i64 %11, %13
  br i1 %14, label %15, label %27

15:                                               ; preds = %2
  %16 = load ptr, ptr @NextPtr, align 8, !tbaa !11
  %17 = ptrtoint ptr %16 to i64
  %18 = load ptr, ptr @Yptr, align 8, !tbaa !14
  %19 = ptrtoint ptr %18 to i64
  %20 = sub i64 %17, %19
  %21 = load ptr, ptr @Aptr, align 8, !tbaa !16
  %22 = ptrtoint ptr %21 to i64
  %23 = sub i64 %19, %22
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i64 noundef %12)
  %25 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i64 noundef %23)
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i64 noundef %20)
  br label %27

27:                                               ; preds = %2, %15
  %28 = phi i32 [ 0, %15 ], [ 1, %2 ]
  ret i32 %28
}

; Function Attrs: nofree nounwind
declare noundef i32 @putchar(i32 noundef) local_unnamed_addr #2

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS4test", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"p2 _ZTS4test", !13, i64 0}
!13 = !{!"any p2 pointer", !8, i64 0}
!14 = !{!15, !15, i64 0}
!15 = !{!"p1 int", !8, i64 0}
!16 = !{!17, !17, i64 0}
!17 = !{!"p1 long", !8, i64 0}
