; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/hash.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Shootout/hash.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [15 x i8] c"malloc ht_node\00", align 1
@.str.1 = private unnamed_addr constant [14 x i8] c"strdup newkey\00", align 1
@ht_prime_list = internal unnamed_addr constant [28 x i64] [i64 53, i64 97, i64 193, i64 389, i64 769, i64 1543, i64 3079, i64 6151, i64 12289, i64 24593, i64 49157, i64 98317, i64 196613, i64 393241, i64 786433, i64 1572869, i64 3145739, i64 6291469, i64 12582917, i64 25165843, i64 50331653, i64 100663319, i64 201326611, i64 402653189, i64 805306457, i64 1610612741, i64 3221225473, i64 4294967291], align 8
@.str.2 = private unnamed_addr constant [3 x i8] c"%x\00", align 1
@.str.3 = private unnamed_addr constant [3 x i8] c"%d\00", align 1
@.str.4 = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1

; Function Attrs: nofree nounwind uwtable
define dso_local noalias nonnull ptr @ht_node_create(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #14
  %3 = icmp eq ptr %2, null
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  tail call void @perror(ptr noundef nonnull @.str) #15
  tail call void @exit(i32 noundef 1) #16
  unreachable

5:                                                ; preds = %1
  %6 = tail call noalias ptr @strdup(ptr noundef %0) #17
  %7 = icmp eq ptr %6, null
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  tail call void @perror(ptr noundef nonnull @.str.1) #15
  tail call void @exit(i32 noundef 1) #16
  unreachable

9:                                                ; preds = %5
  store ptr %6, ptr %2, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store i32 0, ptr %10, align 8, !tbaa !14
  %11 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store ptr null, ptr %11, align 8, !tbaa !15
  ret ptr %2
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #2

; Function Attrs: cold nofree nounwind
declare void @perror(ptr noundef readonly captures(none)) local_unnamed_addr #3

; Function Attrs: nofree noreturn nounwind
declare void @exit(i32 noundef) local_unnamed_addr #4

; Function Attrs: mustprogress nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite)
declare noalias ptr @strdup(ptr noundef readonly captures(none)) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite) uwtable
define dso_local noalias noundef ptr @ht_create(i32 noundef %0) local_unnamed_addr #6 {
  %2 = tail call noalias dereferenceable_or_null(40) ptr @malloc(i64 noundef 40) #14
  %3 = sext i32 %0 to i64
  br label %4

4:                                                ; preds = %4, %1
  %5 = phi i64 [ %9, %4 ], [ 0, %1 ]
  %6 = getelementptr inbounds nuw i64, ptr @ht_prime_list, i64 %5
  %7 = load i64, ptr %6, align 8, !tbaa !16
  %8 = icmp ult i64 %7, %3
  %9 = add nuw nsw i64 %5, 1
  br i1 %8, label %4, label %10, !llvm.loop !18

10:                                               ; preds = %4
  %11 = trunc i64 %7 to i32
  store i32 %11, ptr %2, align 8, !tbaa !20
  %12 = shl i64 %7, 32
  %13 = ashr exact i64 %12, 32
  %14 = tail call noalias ptr @calloc(i64 noundef %13, i64 noundef 8) #18
  %15 = getelementptr inbounds nuw i8, ptr %2, i64 8
  store ptr %14, ptr %15, align 8, !tbaa !24
  %16 = getelementptr inbounds nuw i8, ptr %2, i64 16
  store i32 0, ptr %16, align 8, !tbaa !25
  %17 = getelementptr inbounds nuw i8, ptr %2, i64 24
  store ptr null, ptr %17, align 8, !tbaa !26
  %18 = getelementptr inbounds nuw i8, ptr %2, i64 32
  store i32 0, ptr %18, align 8, !tbaa !27
  ret ptr %2
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) local_unnamed_addr #7

; Function Attrs: nounwind uwtable
define dso_local void @ht_destroy(ptr noundef captures(none) %0) local_unnamed_addr #8 {
  %2 = load i32, ptr %0, align 8, !tbaa !20
  %3 = icmp sgt i32 %2, 0
  br i1 %3, label %4, label %26

4:                                                ; preds = %1
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br label %6

6:                                                ; preds = %4, %21
  %7 = phi i32 [ %2, %4 ], [ %22, %21 ]
  %8 = phi i64 [ 0, %4 ], [ %23, %21 ]
  %9 = load ptr, ptr %5, align 8, !tbaa !24
  %10 = getelementptr inbounds nuw ptr, ptr %9, i64 %8
  %11 = load ptr, ptr %10, align 8, !tbaa !28
  %12 = icmp eq ptr %11, null
  br i1 %12, label %21, label %13

13:                                               ; preds = %6, %13
  %14 = phi ptr [ %16, %13 ], [ %11, %6 ]
  %15 = getelementptr inbounds nuw i8, ptr %14, i64 16
  %16 = load ptr, ptr %15, align 8, !tbaa !15
  %17 = load ptr, ptr %14, align 8, !tbaa !6
  tail call void @free(ptr noundef %17) #17
  tail call void @free(ptr noundef nonnull %14) #17
  %18 = icmp eq ptr %16, null
  br i1 %18, label %19, label %13, !llvm.loop !29

19:                                               ; preds = %13
  %20 = load i32, ptr %0, align 8, !tbaa !20
  br label %21

21:                                               ; preds = %19, %6
  %22 = phi i32 [ %20, %19 ], [ %7, %6 ]
  %23 = add nuw nsw i64 %8, 1
  %24 = sext i32 %22 to i64
  %25 = icmp slt i64 %23, %24
  br i1 %25, label %6, label %26, !llvm.loop !30

26:                                               ; preds = %21, %1
  %27 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %28 = load ptr, ptr %27, align 8, !tbaa !24
  tail call void @free(ptr noundef %28) #17
  tail call void @free(ptr noundef nonnull %0) #17
  ret void
}

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #9

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main(i32 noundef %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #8 {
  %3 = alloca [32 x i8], align 4
  %4 = icmp eq i32 %0, 2
  br i1 %4, label %5, label %10

5:                                                ; preds = %2
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load ptr, ptr %6, align 8, !tbaa !31
  %8 = tail call i64 @strtol(ptr noundef nonnull captures(none) %7, ptr noundef null, i32 noundef 10) #17
  %9 = trunc i64 %8 to i32
  br label %10

10:                                               ; preds = %2, %5
  %11 = phi i32 [ %9, %5 ], [ 3500000, %2 ]
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #17
  %12 = sext i32 %11 to i64
  br label %13

13:                                               ; preds = %13, %10
  %14 = phi i64 [ %18, %13 ], [ 0, %10 ]
  %15 = getelementptr inbounds nuw i64, ptr @ht_prime_list, i64 %14
  %16 = load i64, ptr %15, align 8, !tbaa !16
  %17 = icmp ult i64 %16, %12
  %18 = add nuw nsw i64 %14, 1
  br i1 %17, label %13, label %19, !llvm.loop !18

19:                                               ; preds = %13
  %20 = trunc i64 %16 to i32
  %21 = shl i64 %16, 32
  %22 = ashr exact i64 %21, 32
  %23 = tail call noalias ptr @calloc(i64 noundef %22, i64 noundef 8) #18
  %24 = icmp slt i32 %11, 1
  br i1 %24, label %123, label %25

25:                                               ; preds = %19, %80
  %26 = phi i32 [ %83, %80 ], [ 1, %19 ]
  %27 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) @.str.2, i32 noundef %26) #17
  %28 = load i8, ptr %3, align 4, !tbaa !32
  %29 = icmp eq i8 %28, 0
  br i1 %29, label %40, label %30

30:                                               ; preds = %25, %30
  %31 = phi i8 [ %38, %30 ], [ %28, %25 ]
  %32 = phi i64 [ %36, %30 ], [ 0, %25 ]
  %33 = phi ptr [ %37, %30 ], [ %3, %25 ]
  %34 = mul i64 %32, 5
  %35 = zext i8 %31 to i64
  %36 = add i64 %34, %35
  %37 = getelementptr inbounds nuw i8, ptr %33, i64 1
  %38 = load i8, ptr %37, align 1, !tbaa !32
  %39 = icmp eq i8 %38, 0
  br i1 %39, label %40, label %30, !llvm.loop !33

40:                                               ; preds = %30, %25
  %41 = phi i64 [ 0, %25 ], [ %36, %30 ]
  %42 = urem i64 %41, %22
  %43 = shl i64 %42, 32
  %44 = ashr exact i64 %43, 29
  %45 = getelementptr inbounds i8, ptr %23, i64 %44
  %46 = load ptr, ptr %45, align 8, !tbaa !28
  %47 = icmp eq ptr %46, null
  br i1 %47, label %69, label %48

48:                                               ; preds = %40, %53
  %49 = phi ptr [ %55, %53 ], [ %46, %40 ]
  %50 = load ptr, ptr %49, align 8, !tbaa !6
  %51 = call i32 @strcmp(ptr noundef nonnull readonly dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) %50) #19
  %52 = icmp eq i32 %51, 0
  br i1 %52, label %80, label %53

53:                                               ; preds = %48
  %54 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %55 = load ptr, ptr %54, align 8, !tbaa !28
  %56 = icmp eq ptr %55, null
  br i1 %56, label %57, label %48, !llvm.loop !34

57:                                               ; preds = %53
  %58 = getelementptr inbounds nuw i8, ptr %49, i64 16
  %59 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #14
  %60 = icmp eq ptr %59, null
  br i1 %60, label %61, label %62

61:                                               ; preds = %57
  tail call void @perror(ptr noundef nonnull @.str) #15
  tail call void @exit(i32 noundef 1) #16
  unreachable

62:                                               ; preds = %57
  %63 = call noalias ptr @strdup(ptr noundef nonnull readonly %3) #17
  %64 = icmp eq ptr %63, null
  br i1 %64, label %65, label %66

65:                                               ; preds = %62
  tail call void @perror(ptr noundef nonnull @.str.1) #15
  tail call void @exit(i32 noundef 1) #16
  unreachable

66:                                               ; preds = %62
  store ptr %63, ptr %59, align 8, !tbaa !6
  %67 = getelementptr inbounds nuw i8, ptr %59, i64 8
  store i32 0, ptr %67, align 8, !tbaa !14
  %68 = getelementptr inbounds nuw i8, ptr %59, i64 16
  store ptr null, ptr %68, align 8, !tbaa !15
  store ptr %59, ptr %58, align 8, !tbaa !15
  br label %80

69:                                               ; preds = %40
  %70 = tail call noalias dereferenceable_or_null(24) ptr @malloc(i64 noundef 24) #14
  %71 = icmp eq ptr %70, null
  br i1 %71, label %72, label %73

72:                                               ; preds = %69
  tail call void @perror(ptr noundef nonnull @.str) #15
  tail call void @exit(i32 noundef 1) #16
  unreachable

73:                                               ; preds = %69
  %74 = call noalias ptr @strdup(ptr noundef nonnull readonly %3) #17
  %75 = icmp eq ptr %74, null
  br i1 %75, label %76, label %77

76:                                               ; preds = %73
  tail call void @perror(ptr noundef nonnull @.str.1) #15
  tail call void @exit(i32 noundef 1) #16
  unreachable

77:                                               ; preds = %73
  store ptr %74, ptr %70, align 8, !tbaa !6
  %78 = getelementptr inbounds nuw i8, ptr %70, i64 8
  store i32 0, ptr %78, align 8, !tbaa !14
  %79 = getelementptr inbounds nuw i8, ptr %70, i64 16
  store ptr null, ptr %79, align 8, !tbaa !15
  store ptr %70, ptr %45, align 8, !tbaa !28
  br label %80

80:                                               ; preds = %48, %66, %77
  %81 = phi ptr [ %59, %66 ], [ %70, %77 ], [ %49, %48 ]
  %82 = getelementptr inbounds nuw i8, ptr %81, i64 8
  store i32 %26, ptr %82, align 8, !tbaa !14
  %83 = add nuw i32 %26, 1
  %84 = icmp eq i32 %26, %11
  br i1 %84, label %85, label %25, !llvm.loop !35

85:                                               ; preds = %80, %118
  %86 = phi i32 [ %121, %118 ], [ %11, %80 ]
  %87 = phi i32 [ %120, %118 ], [ 0, %80 ]
  %88 = call i32 (ptr, ptr, ...) @sprintf(ptr noundef nonnull dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) @.str.3, i32 noundef %86) #17
  %89 = load i8, ptr %3, align 4, !tbaa !32
  %90 = icmp eq i8 %89, 0
  br i1 %90, label %101, label %91

91:                                               ; preds = %85, %91
  %92 = phi i8 [ %99, %91 ], [ %89, %85 ]
  %93 = phi i64 [ %97, %91 ], [ 0, %85 ]
  %94 = phi ptr [ %98, %91 ], [ %3, %85 ]
  %95 = mul i64 %93, 5
  %96 = zext i8 %92 to i64
  %97 = add i64 %95, %96
  %98 = getelementptr inbounds nuw i8, ptr %94, i64 1
  %99 = load i8, ptr %98, align 1, !tbaa !32
  %100 = icmp eq i8 %99, 0
  br i1 %100, label %101, label %91, !llvm.loop !33

101:                                              ; preds = %91, %85
  %102 = phi i64 [ 0, %85 ], [ %97, %91 ]
  %103 = urem i64 %102, %22
  %104 = shl i64 %103, 32
  %105 = ashr exact i64 %104, 29
  %106 = getelementptr inbounds i8, ptr %23, i64 %105
  %107 = load ptr, ptr %106, align 8, !tbaa !28
  %108 = icmp eq ptr %107, null
  br i1 %108, label %118, label %109

109:                                              ; preds = %101, %114
  %110 = phi ptr [ %116, %114 ], [ %107, %101 ]
  %111 = load ptr, ptr %110, align 8, !tbaa !6
  %112 = call i32 @strcmp(ptr noundef nonnull readonly dereferenceable(1) %3, ptr noundef nonnull dereferenceable(1) %111) #19
  %113 = icmp eq i32 %112, 0
  br i1 %113, label %118, label %114

114:                                              ; preds = %109
  %115 = getelementptr inbounds nuw i8, ptr %110, i64 16
  %116 = load ptr, ptr %115, align 8, !tbaa !28
  %117 = icmp eq ptr %116, null
  br i1 %117, label %118, label %109, !llvm.loop !36

118:                                              ; preds = %109, %114, %101
  %119 = phi i32 [ 0, %101 ], [ 1, %109 ], [ 0, %114 ]
  %120 = add nuw nsw i32 %119, %87
  %121 = add nsw i32 %86, -1
  %122 = icmp sgt i32 %86, 1
  br i1 %122, label %85, label %123, !llvm.loop !37

123:                                              ; preds = %118, %19
  %124 = phi i32 [ 0, %19 ], [ %120, %118 ]
  %125 = icmp sgt i32 %20, 0
  br i1 %125, label %126, label %143

126:                                              ; preds = %123
  %127 = and i64 %16, 2147483647
  %128 = tail call i64 @llvm.umax.i64(i64 %127, i64 1)
  br label %129

129:                                              ; preds = %126, %140
  %130 = phi i64 [ %141, %140 ], [ 0, %126 ]
  %131 = getelementptr inbounds nuw ptr, ptr %23, i64 %130
  %132 = load ptr, ptr %131, align 8, !tbaa !28
  %133 = icmp eq ptr %132, null
  br i1 %133, label %140, label %134

134:                                              ; preds = %129, %134
  %135 = phi ptr [ %137, %134 ], [ %132, %129 ]
  %136 = getelementptr inbounds nuw i8, ptr %135, i64 16
  %137 = load ptr, ptr %136, align 8, !tbaa !15
  %138 = load ptr, ptr %135, align 8, !tbaa !6
  tail call void @free(ptr noundef %138) #17
  tail call void @free(ptr noundef nonnull %135) #17
  %139 = icmp eq ptr %137, null
  br i1 %139, label %140, label %134, !llvm.loop !29

140:                                              ; preds = %134, %129
  %141 = add nuw nsw i64 %130, 1
  %142 = icmp eq i64 %141, %128
  br i1 %142, label %143, label %129, !llvm.loop !30

143:                                              ; preds = %140, %123
  tail call void @free(ptr noundef %23) #17
  %144 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.4, i32 noundef %124)
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #17
  ret i32 0
}

; Function Attrs: nofree nounwind
declare noundef i32 @sprintf(ptr noalias noundef writeonly captures(none), ptr noundef readonly captures(none), ...) local_unnamed_addr #10

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #10

; Function Attrs: mustprogress nocallback nofree nounwind willreturn
declare i64 @strtol(ptr noundef readonly, ptr noundef captures(none), i32 noundef) local_unnamed_addr #11

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i32 @strcmp(ptr noundef captures(none), ptr noundef captures(none)) local_unnamed_addr #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.umax.i64(i64, i64) #13

attributes #0 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { mustprogress nofree nounwind willreturn memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { nofree nounwind memory(write, argmem: none, inaccessiblemem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #7 = { mustprogress nofree nounwind willreturn allockind("alloc,zeroed") allocsize(0,1) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #8 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #11 = { mustprogress nocallback nofree nounwind willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #12 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #13 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #14 = { nounwind allocsize(0) }
attributes #15 = { cold }
attributes #16 = { cold noreturn nounwind }
attributes #17 = { nounwind }
attributes #18 = { nounwind allocsize(0,1) }
attributes #19 = { nounwind willreturn memory(read) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !8, i64 0}
!7 = !{!"ht_node", !8, i64 0, !12, i64 8, !13, i64 16}
!8 = !{!"p1 omnipotent char", !9, i64 0}
!9 = !{!"any pointer", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C/C++ TBAA"}
!12 = !{!"int", !10, i64 0}
!13 = !{!"p1 _ZTS7ht_node", !9, i64 0}
!14 = !{!7, !12, i64 8}
!15 = !{!7, !13, i64 16}
!16 = !{!17, !17, i64 0}
!17 = !{!"long", !10, i64 0}
!18 = distinct !{!18, !19}
!19 = !{!"llvm.loop.mustprogress"}
!20 = !{!21, !12, i64 0}
!21 = !{!"ht_ht", !12, i64 0, !22, i64 8, !12, i64 16, !13, i64 24, !12, i64 32}
!22 = !{!"p2 _ZTS7ht_node", !23, i64 0}
!23 = !{!"any p2 pointer", !9, i64 0}
!24 = !{!21, !22, i64 8}
!25 = !{!21, !12, i64 16}
!26 = !{!21, !13, i64 24}
!27 = !{!21, !12, i64 32}
!28 = !{!13, !13, i64 0}
!29 = distinct !{!29, !19}
!30 = distinct !{!30, !19}
!31 = !{!8, !8, i64 0}
!32 = !{!10, !10, i64 0}
!33 = distinct !{!33, !19}
!34 = distinct !{!34, !19}
!35 = distinct !{!35, !19}
!36 = distinct !{!36, !19}
!37 = distinct !{!37, !19}
