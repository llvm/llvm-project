; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/general-test.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/UnitTests/Integer/general-test.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.MyStruct = type { ptr, i8, i16, i32, i64 }

@.str = private unnamed_addr constant [24 x i8] c"sizeof(MyStruct) == %d\0A\00", align 1
@.str.1 = private unnamed_addr constant [26 x i8] c"sizeof(My17BitInt) == %d\0A\00", align 1
@.str.2 = private unnamed_addr constant [17 x i8] c"sizeof(j) == %d\0A\00", align 1
@Data1 = dso_local global %struct.MyStruct zeroinitializer, align 8
@Data2 = dso_local global %struct.MyStruct zeroinitializer, align 8
@.str.3 = private unnamed_addr constant [8 x i8] c"j = %d\0A\00", align 1
@.str.4 = private unnamed_addr constant [16 x i8] c"size sum is %d\0A\00", align 1
@.str.5 = private unnamed_addr constant [11 x i8] c"rand = %d\0A\00", align 1
@.str.6 = private unnamed_addr constant [11 x i8] c"argc = %d\0A\00", align 1
@.str.7 = private unnamed_addr constant [11 x i8] c"num  = %d\0A\00", align 1
@.str.8 = private unnamed_addr constant [11 x i8] c"val  = %d\0A\00", align 1
@.str.9 = private unnamed_addr constant [20 x i8] c"that.i4Field  = %d\0A\00", align 1
@.str.10 = private unnamed_addr constant [20 x i8] c"that.i12Field = %d\0A\00", align 1
@.str.11 = private unnamed_addr constant [20 x i8] c"that.i17Field = %d\0A\00", align 1
@.str.12 = private unnamed_addr constant [20 x i8] c"that.i37Field = %d\0A\00", align 1
@.str.13 = private unnamed_addr constant [20 x i8] c"next.i4Field  = %d\0A\00", align 1
@.str.14 = private unnamed_addr constant [20 x i8] c"next.i12Field = %d\0A\00", align 1
@.str.15 = private unnamed_addr constant [20 x i8] c"next.i17Field = %d\0A\00", align 1
@.str.16 = private unnamed_addr constant [20 x i8] c"next.i37Field = %d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef nonnull ptr @getSizes(i16 noundef %0, ptr noundef captures(none) initializes((0, 8)) %1) local_unnamed_addr #0 {
  %3 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef 24)
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i64 noundef 4)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i64 noundef 2)
  store i64 30, ptr %1, align 8, !tbaa !6
  %6 = trunc i16 %0 to i8
  store i8 %6, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 8), align 8, !tbaa !10
  %7 = sext i16 %0 to i32
  %8 = add i16 %0, 1
  store i16 %8, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 10), align 2, !tbaa !16
  %9 = add nsw i32 %7, 2
  store i32 %9, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 12), align 4, !tbaa !17
  %10 = add nsw i32 %7, 3
  %11 = sext i32 %10 to i64
  store i64 %11, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 16), align 8, !tbaa !18
  store ptr null, ptr @Data1, align 8, !tbaa !19
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @Data2, ptr noundef nonnull align 8 dereferenceable(24) @Data1, i64 24, i1 false), !tbaa.struct !20
  %12 = load i8, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 8), align 8, !tbaa !10
  %13 = mul i8 %12, 7
  store i8 %13, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 8), align 8, !tbaa !10
  %14 = load i16, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 10), align 2, !tbaa !16
  %15 = mul i16 %14, 7
  store i16 %15, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 10), align 2, !tbaa !16
  %16 = load i32, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 12), align 4, !tbaa !17
  %17 = mul i32 %16, 7
  store i32 %17, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 12), align 4, !tbaa !17
  %18 = load i64, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 16), align 8, !tbaa !18
  %19 = mul i64 %18, 7
  store i64 %19, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 16), align 8, !tbaa !18
  store ptr @Data1, ptr @Data2, align 8, !tbaa !19
  %20 = shl i16 %0, 1
  %21 = sext i16 %20 to i32
  %22 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %21)
  %23 = load i64, ptr %1, align 8, !tbaa !6
  %24 = trunc i64 %23 to i32
  %25 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %24)
  ret ptr @Data2
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #2

; Function Attrs: nounwind uwtable
define dso_local noundef range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #3 {
  tail call void @srand(i32 noundef 0) #6
  %3 = tail call i32 @rand() #6
  %4 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.5, i32 noundef %3)
  %5 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.6, i32 noundef %0)
  %6 = icmp sgt i32 %0, 1
  br i1 %6, label %7, label %12

7:                                                ; preds = %2
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load ptr, ptr %8, align 8, !tbaa !25
  %10 = tail call i64 @strtol(ptr noundef nonnull captures(none) %9, ptr noundef null, i32 noundef 10) #6
  %11 = trunc i64 %10 to i32
  br label %12

12:                                               ; preds = %7, %2
  %13 = phi i32 [ %11, %7 ], [ 0, %2 ]
  %14 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.7, i32 noundef %13)
  %15 = add nsw i32 %3, %0
  %16 = add nsw i32 %15, %13
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.8, i32 noundef %16)
  %18 = trunc i32 %16 to i16
  %19 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i64 noundef 24)
  %20 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i64 noundef 4)
  %21 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.2, i64 noundef 2)
  %22 = trunc i32 %16 to i8
  store i8 %22, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 8), align 8, !tbaa !10
  %23 = shl i32 %16, 16
  %24 = ashr exact i32 %23, 16
  %25 = add i16 %18, 1
  store i16 %25, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 10), align 2, !tbaa !16
  %26 = add nsw i32 %24, 2
  store i32 %26, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 12), align 4, !tbaa !17
  %27 = add nsw i32 %24, 3
  %28 = sext i32 %27 to i64
  store i64 %28, ptr getelementptr inbounds nuw (i8, ptr @Data1, i64 16), align 8, !tbaa !18
  store ptr null, ptr @Data1, align 8, !tbaa !19
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(24) @Data2, ptr noundef nonnull align 8 dereferenceable(24) @Data1, i64 24, i1 false), !tbaa.struct !20
  %29 = load i8, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 8), align 8, !tbaa !10
  %30 = mul i8 %29, 7
  store i8 %30, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 8), align 8, !tbaa !10
  %31 = load i16, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 10), align 2, !tbaa !16
  %32 = mul i16 %31, 7
  store i16 %32, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 10), align 2, !tbaa !16
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 12), align 4, !tbaa !17
  %34 = mul i32 %33, 7
  store i32 %34, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 12), align 4, !tbaa !17
  %35 = load i64, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 16), align 8, !tbaa !18
  %36 = mul i64 %35, 7
  store i64 %36, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 16), align 8, !tbaa !18
  store ptr @Data1, ptr @Data2, align 8, !tbaa !19
  %37 = shl i16 %18, 1
  %38 = sext i16 %37 to i32
  %39 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %38)
  %40 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef 30)
  %41 = load i8, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 8), align 8, !tbaa !10
  %42 = zext i8 %41 to i32
  %43 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.9, i32 noundef %42)
  %44 = load i16, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 10), align 2, !tbaa !16
  %45 = zext i16 %44 to i32
  %46 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.10, i32 noundef %45)
  %47 = load i32, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 12), align 4, !tbaa !17
  %48 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.11, i32 noundef %47)
  %49 = load i64, ptr getelementptr inbounds nuw (i8, ptr @Data2, i64 16), align 8, !tbaa !18
  %50 = trunc i64 %49 to i32
  %51 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.12, i32 noundef %50)
  %52 = load ptr, ptr @Data2, align 8, !tbaa !19
  %53 = getelementptr inbounds nuw i8, ptr %52, i64 8
  %54 = load i8, ptr %53, align 8, !tbaa !10
  %55 = zext i8 %54 to i32
  %56 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.13, i32 noundef %55)
  %57 = load ptr, ptr @Data2, align 8, !tbaa !19
  %58 = getelementptr inbounds nuw i8, ptr %57, i64 10
  %59 = load i16, ptr %58, align 2, !tbaa !16
  %60 = zext i16 %59 to i32
  %61 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.14, i32 noundef %60)
  %62 = load ptr, ptr @Data2, align 8, !tbaa !19
  %63 = getelementptr inbounds nuw i8, ptr %62, i64 12
  %64 = load i32, ptr %63, align 4, !tbaa !17
  %65 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.15, i32 noundef %64)
  %66 = load ptr, ptr @Data2, align 8, !tbaa !19
  %67 = getelementptr inbounds nuw i8, ptr %66, i64 16
  %68 = load i64, ptr %67, align 8, !tbaa !18
  %69 = trunc i64 %68 to i32
  %70 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.16, i32 noundef %69)
  ret i32 0
}

; Function Attrs: nounwind
declare void @srand(i32 noundef) local_unnamed_addr #4

; Function Attrs: nounwind
declare i32 @rand() local_unnamed_addr #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #5

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
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
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !8, i64 8}
!11 = !{!"MyStruct", !12, i64 0, !8, i64 8, !14, i64 10, !15, i64 12, !7, i64 16}
!12 = !{!"p1 _ZTS8MyStruct", !13, i64 0}
!13 = !{!"any pointer", !8, i64 0}
!14 = !{!"short", !8, i64 0}
!15 = !{!"int", !8, i64 0}
!16 = !{!11, !14, i64 10}
!17 = !{!11, !15, i64 12}
!18 = !{!11, !7, i64 16}
!19 = !{!11, !12, i64 0}
!20 = !{i64 0, i64 8, !21, i64 8, i64 1, !22, i64 10, i64 2, !23, i64 12, i64 4, !24, i64 16, i64 8, !6}
!21 = !{!12, !12, i64 0}
!22 = !{!8, !8, i64 0}
!23 = !{!14, !14, i64 0}
!24 = !{!15, !15, i64 0}
!25 = !{!26, !26, i64 0}
!26 = !{!"p1 omnipotent char", !13, i64 0}
