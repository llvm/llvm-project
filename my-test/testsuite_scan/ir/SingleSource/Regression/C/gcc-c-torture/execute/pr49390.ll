; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49390.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr49390.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%union.anon = type { %struct.V, [48 x i8] }
%struct.V = type { %struct.U, %struct.T }
%struct.U = type { i16, i16 }
%struct.T = type { i32, %struct.S }
%struct.S = type { i32, i32 }

@u = dso_local global %union.anon zeroinitializer, align 4
@v = dso_local global i32 0, align 4
@a = dso_local global %struct.S zeroinitializer, align 8
@b = dso_local local_unnamed_addr global ptr null, align 8

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @foo(i32 noundef %0, ptr noundef readnone captures(address) %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp ne i32 %0, 4
  %6 = icmp ne ptr %1, getelementptr inbounds nuw (i8, ptr @u, i64 4)
  %7 = select i1 %5, i1 true, i1 %6
  br i1 %7, label %8, label %9

8:                                                ; preds = %4
  tail call void @abort() #6
  unreachable

9:                                                ; preds = %4
  %10 = add i32 %3, %2
  store volatile i32 %10, ptr @v, align 4, !tbaa !6
  store volatile i32 16384, ptr @v, align 4, !tbaa !6
  ret void
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #1

; Function Attrs: nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable
define dso_local void @bar(i64 %0) local_unnamed_addr #2 {
  %2 = trunc i64 %0 to i32
  %3 = lshr i64 %0, 32
  %4 = trunc nuw i64 %3 to i32
  store volatile i32 %2, ptr @v, align 4, !tbaa !6
  store volatile i32 %4, ptr @v, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree noinline norecurse nounwind memory(readwrite, argmem: read) uwtable
define dso_local range(i32 -2147483647, -2147483648) i32 @baz(ptr noundef readonly captures(none) %0) local_unnamed_addr #3 {
  %2 = load i32, ptr %0, align 4, !tbaa !10
  store volatile i32 %2, ptr @v, align 4, !tbaa !6
  %3 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %4 = load i32, ptr %3, align 4, !tbaa !12
  store volatile i32 %4, ptr @v, align 4, !tbaa !6
  store volatile i32 0, ptr @v, align 4, !tbaa !6
  %5 = load volatile i32, ptr @v, align 4, !tbaa !6
  %6 = add nsw i32 %5, 1
  ret i32 %6
}

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @test(ptr noundef readonly captures(address_is_null) %0) local_unnamed_addr #0 {
  %2 = alloca %struct.S, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #7
  %3 = load i64, ptr @a, align 8
  store i64 %3, ptr %2, align 8
  %4 = icmp eq ptr %0, null
  %5 = getelementptr inbounds nuw i8, ptr %2, i64 4
  %6 = getelementptr inbounds nuw i8, ptr %0, i64 4
  %7 = lshr i64 %3, 32
  %8 = trunc nuw i64 %7 to i32
  br i1 %4, label %16, label %9

9:                                                ; preds = %1
  %10 = load i32, ptr %6, align 4, !tbaa !12
  %11 = and i32 %10, 8191
  %12 = add nsw i32 %11, -8161
  %13 = icmp ult i32 %12, -8145
  br i1 %13, label %14, label %16

14:                                               ; preds = %9
  %15 = load i32, ptr %0, align 4, !tbaa !10
  tail call void @foo(i32 noundef 1, ptr noundef null, i32 noundef %15, i32 noundef %10)
  br label %16

16:                                               ; preds = %9, %1, %14
  %17 = phi ptr [ %6, %14 ], [ %5, %1 ], [ %6, %9 ]
  %18 = phi ptr [ %0, %14 ], [ %2, %1 ], [ %0, %9 ]
  %19 = call i32 @baz(ptr noundef nonnull %18)
  %20 = icmp eq i32 %19, 0
  br i1 %20, label %63, label %21

21:                                               ; preds = %16
  %22 = load ptr, ptr @b, align 8, !tbaa !13
  %23 = getelementptr inbounds nuw i8, ptr %22, i64 2
  %24 = load i16, ptr %23, align 2, !tbaa !16
  %25 = and i16 %24, 2
  %26 = icmp eq i16 %25, 0
  %27 = select i1 %26, i32 4, i32 32
  %28 = load i32, ptr %17, align 4, !tbaa !12
  %29 = and i32 %28, 8191
  %30 = icmp eq i32 %29, 0
  br i1 %30, label %31, label %33

31:                                               ; preds = %21
  %32 = add i32 %27, %8
  store i32 %32, ptr %5, align 4, !tbaa !12
  br label %37

33:                                               ; preds = %21
  %34 = icmp samesign ult i32 %29, %27
  br i1 %34, label %35, label %37

35:                                               ; preds = %33
  %36 = load i32, ptr %18, align 4, !tbaa !10
  tail call void @foo(i32 noundef 2, ptr noundef null, i32 noundef %36, i32 noundef %28)
  br label %63

37:                                               ; preds = %33, %31
  %38 = phi i32 [ %27, %31 ], [ %29, %33 ]
  %39 = and i16 %24, 1
  %40 = icmp ne i16 %39, 0
  %41 = icmp eq i32 %38, %27
  %42 = select i1 %40, i1 %41, i1 false
  br i1 %42, label %43, label %47

43:                                               ; preds = %37
  %44 = load i64, ptr %18, align 4
  tail call void @bar(i64 %44)
  %45 = load i32, ptr %18, align 4, !tbaa !10
  %46 = load i32, ptr %17, align 4, !tbaa !12
  tail call void @foo(i32 noundef 3, ptr noundef null, i32 noundef %45, i32 noundef %46)
  br label %63

47:                                               ; preds = %37
  %48 = load i32, ptr %17, align 4, !tbaa !12
  %49 = and i32 %48, 8191
  %50 = zext nneg i32 %49 to i64
  %51 = getelementptr inbounds nuw i8, ptr %22, i64 %50
  %52 = getelementptr inbounds nuw i8, ptr %51, i64 4
  %53 = load i32, ptr %52, align 4, !tbaa !19
  %54 = load i32, ptr %18, align 4, !tbaa !10
  %55 = icmp ult i32 %53, %54
  br i1 %55, label %63, label %56

56:                                               ; preds = %47
  %57 = icmp eq i32 %53, %54
  br i1 %57, label %58, label %62

58:                                               ; preds = %56
  %59 = getelementptr inbounds nuw i8, ptr %51, i64 8
  %60 = load i32, ptr %59, align 4, !tbaa !21
  %61 = icmp ult i32 %60, %48
  br i1 %61, label %63, label %62

62:                                               ; preds = %58, %56
  tail call void @foo(i32 noundef 4, ptr noundef nonnull %51, i32 noundef %54, i32 noundef %48)
  br label %63

63:                                               ; preds = %47, %58, %62, %16, %43, %35
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #7
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #5 {
  %1 = tail call ptr asm "", "=r,r,0"(ptr nonnull @a, ptr null) #8, !srcloc !22
  store i32 8192, ptr getelementptr inbounds nuw (i8, ptr @u, i64 8), align 4, !tbaa !23
  store ptr @u, ptr @b, align 8, !tbaa !13
  tail call void @test(ptr noundef %1)
  %2 = load volatile i32, ptr @v, align 4, !tbaa !6
  %3 = icmp eq i32 %2, 16384
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #6
  unreachable

5:                                                ; preds = %0
  ret i32 0
}

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree noinline norecurse nounwind memory(readwrite, argmem: read) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind }
attributes #8 = { nounwind memory(none) }

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
!10 = !{!11, !7, i64 0}
!11 = !{!"S", !7, i64 0, !7, i64 4}
!12 = !{!11, !7, i64 4}
!13 = !{!14, !14, i64 0}
!14 = !{!"p1 omnipotent char", !15, i64 0}
!15 = !{!"any pointer", !8, i64 0}
!16 = !{!17, !18, i64 2}
!17 = !{!"U", !18, i64 0, !18, i64 2}
!18 = !{!"short", !8, i64 0}
!19 = !{!20, !7, i64 4}
!20 = !{!"T", !7, i64 0, !11, i64 4}
!21 = !{!20, !7, i64 8}
!22 = !{i64 1508}
!23 = !{!8, !8, i64 0}
