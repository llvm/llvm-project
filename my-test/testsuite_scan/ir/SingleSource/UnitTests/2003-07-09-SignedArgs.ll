; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-07-09-SignedArgs.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/2003-07-09-SignedArgs.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.__va_list = type { ptr, ptr, ptr, i32, i32 }

@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@.str.1 = private unnamed_addr constant [31 x i8] c"getShort():\09%d %d %d %d %d %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [36 x i8] c"getUnknown():\09%d %d %d %d %d %d %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @passShort(i8 noundef %0, i16 noundef %1) local_unnamed_addr #0 {
  %3 = sext i16 %1 to i32
  %4 = sext i8 %0 to i32
  %5 = trunc i16 %1 to i8
  %6 = add i8 %0, %5
  %7 = sub i8 %5, %0
  %8 = sext i8 %0 to i16
  %9 = mul i16 %1, %8
  %10 = mul nsw i32 %3, %4
  %11 = mul i32 %10, %10
  %12 = icmp eq i8 %0, -128
  %13 = zext i1 %12 to i32
  %14 = sext i8 %6 to i32
  %15 = icmp eq i8 %6, 116
  %16 = zext i1 %15 to i32
  %17 = sext i8 %7 to i32
  %18 = icmp eq i8 %7, 116
  %19 = zext i1 %18 to i32
  %20 = icmp eq i16 %1, -3852
  %21 = zext i1 %20 to i32
  %22 = sext i16 %9 to i32
  %23 = icmp eq i16 %9, -31232
  %24 = zext i1 %23 to i32
  %25 = icmp eq i32 %11, -1708916736
  %26 = zext i1 %25 to i32
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %13, i32 noundef %16, i32 noundef %19, i32 noundef %21, i32 noundef %24, i32 noundef %26)
  %28 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %4, i32 noundef %14, i32 noundef %17, i32 noundef %3, i32 noundef %22, i32 noundef %11)
  %29 = add nsw i32 %3, %4
  %30 = add nsw i32 %29, %22
  %31 = add nsw i32 %30, %14
  %32 = add nsw i32 %31, %17
  %33 = add i32 %32, %11
  %34 = shl i32 %33, 16
  %35 = ashr exact i32 %34, 16
  %36 = tail call i32 (i8, ...) @getUnknown(i8 noundef %0, i32 noundef %14, i32 noundef %17, i32 noundef %3, i32 noundef %22, i32 noundef %35, i32 noundef %11)
  ret i32 %36
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local i16 @getShort(i8 noundef %0, i8 noundef %1, i8 noundef %2, i16 noundef %3, i16 noundef %4, i32 noundef %5) local_unnamed_addr #0 {
  %7 = sext i8 %0 to i32
  %8 = icmp eq i8 %0, -128
  %9 = zext i1 %8 to i32
  %10 = sext i8 %1 to i32
  %11 = icmp eq i8 %1, 116
  %12 = zext i1 %11 to i32
  %13 = sext i8 %2 to i32
  %14 = icmp eq i8 %2, 116
  %15 = zext i1 %14 to i32
  %16 = sext i16 %3 to i32
  %17 = icmp eq i16 %3, -3852
  %18 = zext i1 %17 to i32
  %19 = sext i16 %4 to i32
  %20 = icmp eq i16 %4, -31232
  %21 = zext i1 %20 to i32
  %22 = icmp eq i32 %5, -1708916736
  %23 = zext i1 %22 to i32
  %24 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %9, i32 noundef %12, i32 noundef %15, i32 noundef %18, i32 noundef %21, i32 noundef %23)
  %25 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef %7, i32 noundef %10, i32 noundef %13, i32 noundef %16, i32 noundef %19, i32 noundef %5)
  %26 = add nsw i32 %10, %7
  %27 = add nsw i32 %26, %13
  %28 = add nsw i32 %27, %16
  %29 = add nsw i32 %28, %19
  %30 = add i32 %29, %5
  %31 = trunc i32 %30 to i16
  ret i16 %31
}

; Function Attrs: nofree nounwind uwtable
define dso_local i32 @getUnknown(i8 noundef %0, ...) local_unnamed_addr #0 {
  %2 = alloca %struct.__va_list, align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #4
  call void @llvm.va_start.p0(ptr nonnull %2)
  %3 = getelementptr inbounds nuw i8, ptr %2, i64 24
  %4 = load i32, ptr %3, align 8
  %5 = icmp sgt i32 %4, -1
  br i1 %5, label %9, label %6

6:                                                ; preds = %1
  %7 = add nsw i32 %4, 8
  store i32 %7, ptr %3, align 8
  %8 = icmp samesign ult i32 %4, -7
  br i1 %8, label %13, label %9

9:                                                ; preds = %1, %6
  %10 = load ptr, ptr %2, align 8
  %11 = getelementptr inbounds nuw i8, ptr %10, i64 8
  store ptr %11, ptr %2, align 8
  %12 = load i32, ptr %10, align 8, !tbaa !6
  br label %23

13:                                               ; preds = %6
  %14 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %15 = load ptr, ptr %14, align 8
  %16 = sext i32 %4 to i64
  %17 = getelementptr inbounds i8, ptr %15, i64 %16
  %18 = load i32, ptr %17, align 8, !tbaa !6
  %19 = icmp sgt i32 %4, -9
  br i1 %19, label %23, label %20

20:                                               ; preds = %13
  %21 = add nsw i32 %4, 16
  store i32 %21, ptr %3, align 8
  %22 = icmp samesign ult i32 %7, -7
  br i1 %22, label %28, label %23

23:                                               ; preds = %13, %20, %9
  %24 = phi i32 [ %18, %20 ], [ %18, %13 ], [ %12, %9 ]
  %25 = load ptr, ptr %2, align 8
  %26 = getelementptr inbounds nuw i8, ptr %25, i64 8
  store ptr %26, ptr %2, align 8
  %27 = load i32, ptr %25, align 8, !tbaa !6
  br label %38

28:                                               ; preds = %20
  %29 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %30 = load ptr, ptr %29, align 8
  %31 = sext i32 %7 to i64
  %32 = getelementptr inbounds i8, ptr %30, i64 %31
  %33 = load i32, ptr %32, align 8, !tbaa !6
  %34 = icmp sgt i32 %4, -17
  br i1 %34, label %38, label %35

35:                                               ; preds = %28
  %36 = add nsw i32 %4, 24
  store i32 %36, ptr %3, align 8
  %37 = icmp samesign ult i32 %21, -7
  br i1 %37, label %44, label %38

38:                                               ; preds = %28, %35, %23
  %39 = phi i32 [ %33, %35 ], [ %33, %28 ], [ %27, %23 ]
  %40 = phi i32 [ %18, %35 ], [ %18, %28 ], [ %24, %23 ]
  %41 = load ptr, ptr %2, align 8
  %42 = getelementptr inbounds nuw i8, ptr %41, i64 8
  store ptr %42, ptr %2, align 8
  %43 = load i32, ptr %41, align 8, !tbaa !6
  br label %54

44:                                               ; preds = %35
  %45 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %46 = load ptr, ptr %45, align 8
  %47 = sext i32 %21 to i64
  %48 = getelementptr inbounds i8, ptr %46, i64 %47
  %49 = load i32, ptr %48, align 8, !tbaa !6
  %50 = icmp sgt i32 %4, -25
  br i1 %50, label %54, label %51

51:                                               ; preds = %44
  %52 = add nsw i32 %4, 32
  store i32 %52, ptr %3, align 8
  %53 = icmp samesign ult i32 %36, -7
  br i1 %53, label %61, label %54

54:                                               ; preds = %44, %51, %38
  %55 = phi i32 [ %49, %51 ], [ %49, %44 ], [ %43, %38 ]
  %56 = phi i32 [ %18, %51 ], [ %18, %44 ], [ %40, %38 ]
  %57 = phi i32 [ %33, %51 ], [ %33, %44 ], [ %39, %38 ]
  %58 = load ptr, ptr %2, align 8
  %59 = getelementptr inbounds nuw i8, ptr %58, i64 8
  store ptr %59, ptr %2, align 8
  %60 = load i32, ptr %58, align 8, !tbaa !6
  br label %71

61:                                               ; preds = %51
  %62 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %63 = load ptr, ptr %62, align 8
  %64 = sext i32 %36 to i64
  %65 = getelementptr inbounds i8, ptr %63, i64 %64
  %66 = load i32, ptr %65, align 8, !tbaa !6
  %67 = icmp sgt i32 %4, -33
  br i1 %67, label %71, label %68

68:                                               ; preds = %61
  %69 = add nsw i32 %4, 40
  store i32 %69, ptr %3, align 8
  %70 = icmp samesign ult i32 %52, -7
  br i1 %70, label %79, label %71

71:                                               ; preds = %61, %68, %54
  %72 = phi i32 [ %66, %68 ], [ %66, %61 ], [ %60, %54 ]
  %73 = phi i32 [ %33, %68 ], [ %33, %61 ], [ %57, %54 ]
  %74 = phi i32 [ %18, %68 ], [ %18, %61 ], [ %56, %54 ]
  %75 = phi i32 [ %49, %68 ], [ %49, %61 ], [ %55, %54 ]
  %76 = load ptr, ptr %2, align 8
  %77 = getelementptr inbounds nuw i8, ptr %76, i64 8
  store ptr %77, ptr %2, align 8
  %78 = load i32, ptr %76, align 8, !tbaa !6
  br label %94

79:                                               ; preds = %68
  %80 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %81 = load ptr, ptr %80, align 8
  %82 = sext i32 %52 to i64
  %83 = getelementptr inbounds i8, ptr %81, i64 %82
  %84 = load i32, ptr %83, align 8, !tbaa !6
  %85 = icmp sgt i32 %4, -41
  br i1 %85, label %94, label %86

86:                                               ; preds = %79
  %87 = add nsw i32 %4, 48
  store i32 %87, ptr %3, align 8
  %88 = icmp samesign ult i32 %69, -7
  br i1 %88, label %89, label %94

89:                                               ; preds = %86
  %90 = getelementptr inbounds nuw i8, ptr %2, i64 8
  %91 = load ptr, ptr %90, align 8
  %92 = sext i32 %69 to i64
  %93 = getelementptr inbounds i8, ptr %91, i64 %92
  br label %102

94:                                               ; preds = %71, %86, %79
  %95 = phi i32 [ %78, %71 ], [ %84, %86 ], [ %84, %79 ]
  %96 = phi i32 [ %75, %71 ], [ %49, %86 ], [ %49, %79 ]
  %97 = phi i32 [ %74, %71 ], [ %18, %86 ], [ %18, %79 ]
  %98 = phi i32 [ %73, %71 ], [ %33, %86 ], [ %33, %79 ]
  %99 = phi i32 [ %72, %71 ], [ %66, %86 ], [ %66, %79 ]
  %100 = load ptr, ptr %2, align 8
  %101 = getelementptr inbounds nuw i8, ptr %100, i64 8
  store ptr %101, ptr %2, align 8
  br label %102

102:                                              ; preds = %94, %89
  %103 = phi i32 [ %84, %89 ], [ %95, %94 ]
  %104 = phi i32 [ %49, %89 ], [ %96, %94 ]
  %105 = phi i32 [ %18, %89 ], [ %97, %94 ]
  %106 = phi i32 [ %33, %89 ], [ %98, %94 ]
  %107 = phi i32 [ %66, %89 ], [ %99, %94 ]
  %108 = phi ptr [ %93, %89 ], [ %100, %94 ]
  %109 = load i32, ptr %108, align 8, !tbaa !6
  call void @llvm.va_end.p0(ptr nonnull %2)
  %110 = sext i8 %0 to i32
  %111 = shl i32 %105, 24
  %112 = ashr exact i32 %111, 24
  %113 = shl i32 %106, 24
  %114 = ashr exact i32 %113, 24
  %115 = shl i32 %104, 16
  %116 = ashr exact i32 %115, 16
  %117 = shl i32 %107, 16
  %118 = ashr exact i32 %117, 16
  %119 = shl i32 %103, 16
  %120 = ashr exact i32 %119, 16
  %121 = call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %110, i32 noundef %112, i32 noundef %114, i32 noundef %116, i32 noundef %118, i32 noundef %120, i32 noundef %109)
  %122 = add nsw i32 %112, %110
  %123 = add nsw i32 %122, %114
  %124 = add nsw i32 %123, %116
  %125 = add nsw i32 %124, %118
  %126 = add nsw i32 %125, %120
  %127 = add nsw i32 %126, %109
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #4
  ret i32 %127
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #0 {
  %1 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1, i32 noundef 1)
  %2 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef -128, i32 noundef 116, i32 noundef 116, i32 noundef -3852, i32 noundef -31232, i32 noundef -1708916736)
  %3 = tail call i32 (i8, ...) @getUnknown(i8 noundef -128, i32 noundef 116, i32 noundef 116, i32 noundef -3852, i32 noundef -31232, i32 noundef 30556, i32 noundef -1708916736)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %3)
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn
declare void @llvm.va_end.p0(ptr) #3

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind willreturn }
attributes #4 = { nounwind }

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
