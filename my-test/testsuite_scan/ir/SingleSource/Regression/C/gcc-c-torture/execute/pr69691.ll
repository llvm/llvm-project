; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr69691.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr69691.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.S = type { [10 x i8], [31 x ptr] }

@u = dso_local global [6 x i8] c".ach4\00", align 1
@v = dso_local global [2 x ptr] [ptr @u, ptr null], align 8
@r = dso_local global [7 x %struct.S] zeroinitializer, align 8
@r2 = dso_local local_unnamed_addr global ptr @r, align 8
@.str = private unnamed_addr constant [8 x i8] c"foo %d\0A\00", align 1
@w = internal unnamed_addr global ptr null, align 8
@__const.main.c = private unnamed_addr constant [6 x i8] c"aaaaa\00", align 1

; Function Attrs: nofree noinline nounwind uwtable
define dso_local noundef i32 @fn(i32 noundef returned %0) local_unnamed_addr #0 {
  %2 = tail call ptr @strchr(ptr noundef nonnull dereferenceable(1) @u, i32 noundef %0) #11
  %3 = icmp ne ptr %2, null
  %4 = icmp eq i32 %0, 96
  %5 = or i1 %4, %3
  br i1 %5, label %6, label %7

6:                                                ; preds = %1
  ret i32 %0

7:                                                ; preds = %1
  tail call void @abort() #12
  unreachable
}

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare ptr @strchr(ptr noundef, i32 noundef) local_unnamed_addr #1

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #2

; Function Attrs: nofree noinline nounwind uwtable
define dso_local range(i32 -96, 160) i32 @foo(i8 noundef %0) local_unnamed_addr #0 {
  %2 = zext i8 %0 to i32
  %3 = icmp eq i8 %0, 0
  br i1 %3, label %4, label %5

4:                                                ; preds = %1
  tail call void @abort() #12
  unreachable

5:                                                ; preds = %1
  %6 = tail call i32 @fn(i32 noundef %2)
  %7 = icmp ugt i8 %0, 95
  br i1 %7, label %8, label %14

8:                                                ; preds = %5
  %9 = tail call i32 @fn(i32 noundef %2)
  %10 = icmp ult i8 %0, 123
  br i1 %10, label %11, label %16

11:                                               ; preds = %8
  %12 = tail call i32 @fn(i32 noundef %2)
  %13 = add nsw i32 %2, -96
  br label %18

14:                                               ; preds = %5
  %15 = icmp eq i8 %0, 46
  br i1 %15, label %18, label %16

16:                                               ; preds = %8, %14
  %17 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %2) #11
  br label %18

18:                                               ; preds = %14, %16, %11
  %19 = phi i32 [ %13, %11 ], [ -1, %16 ], [ 0, %14 ]
  ret i32 %19
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nofree noinline nounwind uwtable
define dso_local void @bar(ptr noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = alloca [500 x i8], align 1
  %3 = alloca [10 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #11
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #11
  %4 = load ptr, ptr @r2, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %4, i64 264
  store ptr %5, ptr @r2, align 8, !tbaa !6
  store ptr %4, ptr @w, align 8, !tbaa !6
  %6 = load ptr, ptr %0, align 8, !tbaa !11
  %7 = icmp eq ptr %6, null
  br i1 %7, label %63, label %8

8:                                                ; preds = %1, %59
  %9 = phi ptr [ %61, %59 ], [ %6, %1 ]
  %10 = phi ptr [ %60, %59 ], [ %0, %1 ]
  %11 = call ptr @strcpy(ptr noundef nonnull dereferenceable(1) %2, ptr noundef nonnull dereferenceable(1) %9) #11
  br label %12

12:                                               ; preds = %56, %8
  %13 = phi ptr [ %2, %8 ], [ %58, %56 ]
  %14 = call ptr @strchr(ptr noundef nonnull dereferenceable(1) %13, i32 noundef 32) #11
  %15 = icmp eq ptr %14, null
  br i1 %15, label %17, label %16

16:                                               ; preds = %12
  store i8 0, ptr %14, align 1, !tbaa !13
  br label %17

17:                                               ; preds = %16, %12
  %18 = call i64 @strlen(ptr noundef nonnull dereferenceable(1) %13) #11
  %19 = trunc i64 %18 to i32
  %20 = load ptr, ptr @w, align 8, !tbaa !6
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 1 dereferenceable(10) %3, i8 0, i64 10, i1 false)
  %21 = icmp sgt i32 %19, 0
  br i1 %21, label %22, label %56

22:                                               ; preds = %17
  %23 = and i64 %18, 2147483647
  br label %24

24:                                               ; preds = %22, %51
  %25 = phi i64 [ 0, %22 ], [ %54, %51 ]
  %26 = phi ptr [ %20, %22 ], [ %53, %51 ]
  %27 = phi i32 [ 0, %22 ], [ %52, %51 ]
  %28 = getelementptr inbounds nuw i8, ptr %13, i64 %25
  %29 = load i8, ptr %28, align 1, !tbaa !13
  %30 = add i8 %29, -48
  %31 = icmp ult i8 %30, 10
  br i1 %31, label %32, label %35

32:                                               ; preds = %24
  %33 = sext i32 %27 to i64
  %34 = getelementptr inbounds i8, ptr %3, i64 %33
  store i8 %30, ptr %34, align 1, !tbaa !13
  br label %51

35:                                               ; preds = %24
  %36 = call i32 @foo(i8 noundef %29)
  %37 = getelementptr inbounds nuw i8, ptr %26, i64 16
  %38 = sext i32 %36 to i64
  %39 = getelementptr inbounds ptr, ptr %37, i64 %38
  %40 = load ptr, ptr %39, align 8, !tbaa !6
  %41 = icmp eq ptr %40, null
  br i1 %41, label %42, label %48

42:                                               ; preds = %35
  %43 = load ptr, ptr @r2, align 8, !tbaa !6
  %44 = getelementptr inbounds nuw i8, ptr %43, i64 264
  store ptr %44, ptr @r2, align 8, !tbaa !6
  store ptr %43, ptr %39, align 8, !tbaa !6
  %45 = load ptr, ptr @r2, align 8, !tbaa !6
  %46 = icmp eq ptr %45, getelementptr inbounds nuw (i8, ptr @r, i64 1848)
  br i1 %46, label %47, label %48

47:                                               ; preds = %42
  call void @abort() #12
  unreachable

48:                                               ; preds = %42, %35
  %49 = phi ptr [ %43, %42 ], [ %40, %35 ]
  %50 = add nsw i32 %27, 1
  br label %51

51:                                               ; preds = %32, %48
  %52 = phi i32 [ %27, %32 ], [ %50, %48 ]
  %53 = phi ptr [ %26, %32 ], [ %49, %48 ]
  %54 = add nuw nsw i64 %25, 1
  %55 = icmp eq i64 %54, %23
  br i1 %55, label %56, label %24, !llvm.loop !14

56:                                               ; preds = %51, %17
  %57 = phi ptr [ %20, %17 ], [ %53, %51 ]
  call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(10) %57, ptr noundef nonnull align 1 dereferenceable(10) %3, i64 10, i1 false)
  %58 = getelementptr inbounds nuw i8, ptr %14, i64 1
  br i1 %15, label %59, label %12, !llvm.loop !16

59:                                               ; preds = %56
  %60 = getelementptr inbounds nuw i8, ptr %10, i64 8
  %61 = load ptr, ptr %60, align 8, !tbaa !11
  %62 = icmp eq ptr %61, null
  br i1 %62, label %63, label %8, !llvm.loop !17

63:                                               ; preds = %59, %1
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #11
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #11
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare ptr @strcpy(ptr noalias noundef returned writeonly, ptr noalias noundef readonly captures(none)) local_unnamed_addr #5

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: read)
declare i64 @strlen(ptr noundef captures(none)) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #6

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #7

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

; Function Attrs: noinline nounwind uwtable
define dso_local void @baz(ptr noundef readonly captures(none) %0) local_unnamed_addr #8 {
  %2 = alloca [300 x i8], align 4
  %3 = alloca [300 x i8], align 1
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #11
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #11
  %4 = tail call i64 @strlen(ptr noundef nonnull dereferenceable(1) %0) #11
  %5 = trunc i64 %4 to i32
  store i8 96, ptr %2, align 4, !tbaa !13
  %6 = tail call i32 @llvm.smax.i32(i32 %5, i32 0)
  %7 = zext nneg i32 %6 to i64
  br label %8

8:                                                ; preds = %11, %1
  %9 = phi i64 [ %16, %11 ], [ 0, %1 ]
  %10 = icmp eq i64 %9, %7
  br i1 %10, label %20, label %11

11:                                               ; preds = %8
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 %9
  %13 = load i8, ptr %12, align 1, !tbaa !13
  %14 = zext i8 %13 to i32
  %15 = tail call i32 @fn(i32 noundef %14)
  %16 = add nuw nsw i64 %9, 1
  %17 = getelementptr inbounds nuw i8, ptr %2, i64 %16
  store i8 %13, ptr %17, align 1, !tbaa !13
  %18 = tail call i32 @foo(i8 noundef %13)
  %19 = icmp slt i32 %18, 1
  br i1 %19, label %131, label %8, !llvm.loop !18

20:                                               ; preds = %8
  %21 = shl i64 %4, 32
  %22 = add i64 %21, 4294967296
  %23 = ashr exact i64 %22, 32
  %24 = getelementptr inbounds i8, ptr %2, i64 %23
  store i8 96, ptr %24, align 1, !tbaa !13
  %25 = add i64 %21, 17179869184
  %26 = ashr exact i64 %25, 32
  call void @llvm.memset.p0.i64(ptr nonnull align 1 %3, i8 0, i64 %26, i1 false)
  %27 = load ptr, ptr @w, align 8, !tbaa !6
  %28 = icmp ne ptr %27, null
  %29 = icmp sgt i32 %5, -2
  %30 = and i1 %28, %29
  br i1 %30, label %31, label %131

31:                                               ; preds = %20
  %32 = add i32 %5, 2
  %33 = sext i32 %32 to i64
  %34 = tail call i32 @llvm.smax.i32(i32 %32, i32 1)
  %35 = zext nneg i32 %34 to i64
  br label %40

36:                                               ; preds = %118
  %37 = icmp sgt i32 %5, 3
  br i1 %37, label %38, label %131

38:                                               ; preds = %36
  %39 = and i64 %4, 2147483647
  br label %121

40:                                               ; preds = %31, %118
  %41 = phi i64 [ 0, %31 ], [ %119, %118 ]
  %42 = load ptr, ptr @w, align 8, !tbaa !6
  %43 = getelementptr inbounds nuw i8, ptr %3, i64 %41
  br label %44

44:                                               ; preds = %40, %113
  %45 = phi i64 [ 0, %40 ], [ %117, %113 ]
  %46 = phi i64 [ %41, %40 ], [ %114, %113 ]
  %47 = phi i64 [ 3, %40 ], [ %116, %113 ]
  %48 = phi ptr [ %42, %40 ], [ %56, %113 ]
  %49 = add i64 %45, 3
  %50 = getelementptr inbounds nuw i8, ptr %48, i64 16
  %51 = getelementptr inbounds nuw i8, ptr %2, i64 %46
  %52 = load i8, ptr %51, align 1, !tbaa !13
  %53 = tail call i32 @foo(i8 noundef %52)
  %54 = sext i32 %53 to i64
  %55 = getelementptr inbounds ptr, ptr %50, i64 %54
  %56 = load ptr, ptr %55, align 8, !tbaa !6
  %57 = icmp eq ptr %56, null
  br i1 %57, label %118, label %58

58:                                               ; preds = %44
  %59 = sub nuw nsw i64 %46, %41
  %60 = trunc i64 %59 to i32
  %61 = add i32 %60, 2
  %62 = icmp slt i32 %61, 0
  br i1 %62, label %113, label %63

63:                                               ; preds = %58
  %64 = icmp ult i64 %49, 8
  br i1 %64, label %102, label %65

65:                                               ; preds = %63
  %66 = icmp ult i64 %49, 32
  br i1 %66, label %88, label %67

67:                                               ; preds = %65
  %68 = and i64 %49, -32
  br label %69

69:                                               ; preds = %69, %67
  %70 = phi i64 [ 0, %67 ], [ %81, %69 ]
  %71 = getelementptr inbounds nuw i8, ptr %56, i64 %70
  %72 = getelementptr inbounds nuw i8, ptr %71, i64 16
  %73 = load <16 x i8>, ptr %71, align 1, !tbaa !13
  %74 = load <16 x i8>, ptr %72, align 1, !tbaa !13
  %75 = getelementptr inbounds nuw i8, ptr %43, i64 %70
  %76 = getelementptr inbounds nuw i8, ptr %75, i64 16
  %77 = load <16 x i8>, ptr %75, align 1, !tbaa !13
  %78 = load <16 x i8>, ptr %76, align 1, !tbaa !13
  %79 = tail call <16 x i8> @llvm.umax.v16i8(<16 x i8> %73, <16 x i8> %77)
  %80 = tail call <16 x i8> @llvm.umax.v16i8(<16 x i8> %74, <16 x i8> %78)
  store <16 x i8> %79, ptr %75, align 1
  store <16 x i8> %80, ptr %76, align 1
  %81 = add nuw i64 %70, 32
  %82 = icmp eq i64 %81, %68
  br i1 %82, label %83, label %69, !llvm.loop !19

83:                                               ; preds = %69
  %84 = icmp eq i64 %49, %68
  br i1 %84, label %113, label %85

85:                                               ; preds = %83
  %86 = and i64 %49, 24
  %87 = icmp eq i64 %86, 0
  br i1 %87, label %102, label %88

88:                                               ; preds = %85, %65
  %89 = phi i64 [ %68, %85 ], [ 0, %65 ]
  %90 = and i64 %49, -8
  br label %91

91:                                               ; preds = %91, %88
  %92 = phi i64 [ %89, %88 ], [ %98, %91 ]
  %93 = getelementptr inbounds nuw i8, ptr %56, i64 %92
  %94 = load <8 x i8>, ptr %93, align 1, !tbaa !13
  %95 = getelementptr inbounds nuw i8, ptr %43, i64 %92
  %96 = load <8 x i8>, ptr %95, align 1, !tbaa !13
  %97 = tail call <8 x i8> @llvm.umax.v8i8(<8 x i8> %94, <8 x i8> %96)
  store <8 x i8> %97, ptr %95, align 1
  %98 = add nuw i64 %92, 8
  %99 = icmp eq i64 %98, %90
  br i1 %99, label %100, label %91, !llvm.loop !22

100:                                              ; preds = %91
  %101 = icmp eq i64 %49, %90
  br i1 %101, label %113, label %102

102:                                              ; preds = %85, %100, %63
  %103 = phi i64 [ 0, %63 ], [ %68, %85 ], [ %90, %100 ]
  br label %104

104:                                              ; preds = %102, %104
  %105 = phi i64 [ %111, %104 ], [ %103, %102 ]
  %106 = getelementptr inbounds nuw i8, ptr %56, i64 %105
  %107 = load i8, ptr %106, align 1, !tbaa !13
  %108 = getelementptr inbounds nuw i8, ptr %43, i64 %105
  %109 = load i8, ptr %108, align 1, !tbaa !13
  %110 = tail call i8 @llvm.umax.i8(i8 %107, i8 %109)
  store i8 %110, ptr %108, align 1
  %111 = add nuw nsw i64 %105, 1
  %112 = icmp eq i64 %111, %47
  br i1 %112, label %113, label %104, !llvm.loop !23

113:                                              ; preds = %104, %83, %100, %58
  %114 = add nuw nsw i64 %46, 1
  %115 = icmp slt i64 %114, %33
  %116 = add nuw nsw i64 %47, 1
  %117 = add i64 %45, 1
  br i1 %115, label %44, label %118, !llvm.loop !24

118:                                              ; preds = %113, %44
  %119 = add nuw nsw i64 %41, 1
  %120 = icmp eq i64 %119, %35
  br i1 %120, label %36, label %40, !llvm.loop !25

121:                                              ; preds = %38, %128
  %122 = phi i64 [ 3, %38 ], [ %129, %128 ]
  %123 = getelementptr inbounds nuw i8, ptr %3, i64 %122
  %124 = load i8, ptr %123, align 1, !tbaa !13
  %125 = and i8 %124, 1
  %126 = icmp eq i8 %125, 0
  br i1 %126, label %128, label %127

127:                                              ; preds = %121
  tail call void asm sideeffect "", ""() #11, !srcloc !26
  br label %128

128:                                              ; preds = %121, %127
  %129 = add nuw nsw i64 %122, 1
  %130 = icmp eq i64 %129, %39
  br i1 %130, label %131, label %121, !llvm.loop !27

131:                                              ; preds = %11, %128, %36, %20
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #11
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #11
  ret void
}

; Function Attrs: nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #9 {
  tail call void @bar(ptr noundef nonnull @v)
  tail call void @baz(ptr noundef nonnull @__const.main.c)
  ret i32 0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i8 @llvm.umax.i8(i8, i8) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smax.i32(i32, i32) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x i8> @llvm.umax.v16i8(<16 x i8>, <16 x i8>) #10

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <8 x i8> @llvm.umax.v8i8(<8 x i8>, <8 x i8>) #10

attributes #0 = { nofree noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: read) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #6 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #7 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #8 = { noinline nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #9 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #10 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #11 = { nounwind }
attributes #12 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 _ZTS1S", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"p1 omnipotent char", !8, i64 0}
!13 = !{!9, !9, i64 0}
!14 = distinct !{!14, !15}
!15 = !{!"llvm.loop.mustprogress"}
!16 = distinct !{!16, !15}
!17 = distinct !{!17, !15}
!18 = distinct !{!18, !15}
!19 = distinct !{!19, !15, !20, !21}
!20 = !{!"llvm.loop.isvectorized", i32 1}
!21 = !{!"llvm.loop.unroll.runtime.disable"}
!22 = distinct !{!22, !15, !20, !21}
!23 = distinct !{!23, !15, !21, !20}
!24 = distinct !{!24, !15}
!25 = distinct !{!25, !15}
!26 = !{i64 1976}
!27 = distinct !{!27, !15}
