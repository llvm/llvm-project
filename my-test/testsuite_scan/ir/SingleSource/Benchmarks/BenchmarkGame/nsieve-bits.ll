; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/nsieve-bits.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/BenchmarkGame/nsieve-bits.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@.str = private unnamed_addr constant [22 x i8] c"Primes up to %8d %8d\0A\00", align 1

; Function Attrs: nounwind uwtable
define dso_local range(i32 0, 2) i32 @main(i32 noundef %0, ptr noundef readnone captures(none) %1) local_unnamed_addr #0 {
  %3 = tail call noalias dereferenceable_or_null(5120004) ptr @malloc(i64 noundef 5120004) #5
  %4 = icmp eq ptr %3, null
  br i1 %4, label %117, label %5

5:                                                ; preds = %2
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(5120004) %3, i8 -1, i64 5120004, i1 false)
  br label %6

6:                                                ; preds = %5, %37
  %7 = phi i32 [ 0, %5 ], [ %38, %37 ]
  %8 = phi i32 [ 2, %5 ], [ %39, %37 ]
  %9 = lshr i32 %8, 5
  %10 = zext nneg i32 %9 to i64
  %11 = getelementptr inbounds nuw i32, ptr %3, i64 %10
  %12 = load i32, ptr %11, align 4, !tbaa !6
  %13 = and i32 %8, 31
  %14 = shl nuw i32 1, %13
  %15 = and i32 %12, %14
  %16 = icmp eq i32 %15, 0
  br i1 %16, label %37, label %17

17:                                               ; preds = %6
  %18 = add i32 %7, 1
  %19 = icmp samesign ugt i32 %8, 20480000
  br i1 %19, label %37, label %20

20:                                               ; preds = %17
  %21 = shl nuw nsw i32 %8, 1
  br label %22

22:                                               ; preds = %20, %34
  %23 = phi i32 [ %35, %34 ], [ %21, %20 ]
  %24 = lshr i32 %23, 5
  %25 = zext nneg i32 %24 to i64
  %26 = getelementptr inbounds nuw i32, ptr %3, i64 %25
  %27 = load i32, ptr %26, align 4, !tbaa !6
  %28 = and i32 %23, 31
  %29 = shl nuw i32 1, %28
  %30 = and i32 %27, %29
  %31 = icmp eq i32 %30, 0
  br i1 %31, label %34, label %32

32:                                               ; preds = %22
  %33 = xor i32 %27, %29
  store i32 %33, ptr %26, align 4, !tbaa !6
  br label %34

34:                                               ; preds = %22, %32
  %35 = add nuw nsw i32 %23, %8
  %36 = icmp samesign ugt i32 %35, 40960000
  br i1 %36, label %37, label %22, !llvm.loop !10

37:                                               ; preds = %34, %17, %6
  %38 = phi i32 [ %7, %6 ], [ %18, %17 ], [ %18, %34 ]
  %39 = add nuw nsw i32 %8, 1
  %40 = icmp eq i32 %39, 40960001
  br i1 %40, label %41, label %6, !llvm.loop !12

41:                                               ; preds = %37
  %42 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 40960000, i32 noundef %38)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(2560004) %3, i8 -1, i64 2560004, i1 false)
  br label %43

43:                                               ; preds = %74, %41
  %44 = phi i32 [ 0, %41 ], [ %75, %74 ]
  %45 = phi i32 [ 2, %41 ], [ %76, %74 ]
  %46 = lshr i32 %45, 5
  %47 = zext nneg i32 %46 to i64
  %48 = getelementptr inbounds nuw i32, ptr %3, i64 %47
  %49 = load i32, ptr %48, align 4, !tbaa !6
  %50 = and i32 %45, 31
  %51 = shl nuw i32 1, %50
  %52 = and i32 %49, %51
  %53 = icmp eq i32 %52, 0
  br i1 %53, label %74, label %54

54:                                               ; preds = %43
  %55 = add i32 %44, 1
  %56 = icmp samesign ugt i32 %45, 10240000
  br i1 %56, label %74, label %57

57:                                               ; preds = %54
  %58 = shl nuw nsw i32 %45, 1
  br label %59

59:                                               ; preds = %71, %57
  %60 = phi i32 [ %72, %71 ], [ %58, %57 ]
  %61 = lshr i32 %60, 5
  %62 = zext nneg i32 %61 to i64
  %63 = getelementptr inbounds nuw i32, ptr %3, i64 %62
  %64 = load i32, ptr %63, align 4, !tbaa !6
  %65 = and i32 %60, 31
  %66 = shl nuw i32 1, %65
  %67 = and i32 %64, %66
  %68 = icmp eq i32 %67, 0
  br i1 %68, label %71, label %69

69:                                               ; preds = %59
  %70 = xor i32 %64, %66
  store i32 %70, ptr %63, align 4, !tbaa !6
  br label %71

71:                                               ; preds = %69, %59
  %72 = add nuw nsw i32 %60, %45
  %73 = icmp samesign ugt i32 %72, 20480000
  br i1 %73, label %74, label %59, !llvm.loop !10

74:                                               ; preds = %71, %54, %43
  %75 = phi i32 [ %44, %43 ], [ %55, %54 ], [ %55, %71 ]
  %76 = add nuw nsw i32 %45, 1
  %77 = icmp eq i32 %76, 20480001
  br i1 %77, label %78, label %43, !llvm.loop !12

78:                                               ; preds = %74
  %79 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 20480000, i32 noundef %75)
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(1280004) %3, i8 -1, i64 1280004, i1 false)
  br label %80

80:                                               ; preds = %111, %78
  %81 = phi i32 [ 0, %78 ], [ %112, %111 ]
  %82 = phi i32 [ 2, %78 ], [ %113, %111 ]
  %83 = lshr i32 %82, 5
  %84 = zext nneg i32 %83 to i64
  %85 = getelementptr inbounds nuw i32, ptr %3, i64 %84
  %86 = load i32, ptr %85, align 4, !tbaa !6
  %87 = and i32 %82, 31
  %88 = shl nuw i32 1, %87
  %89 = and i32 %86, %88
  %90 = icmp eq i32 %89, 0
  br i1 %90, label %111, label %91

91:                                               ; preds = %80
  %92 = add i32 %81, 1
  %93 = icmp samesign ugt i32 %82, 5120000
  br i1 %93, label %111, label %94

94:                                               ; preds = %91
  %95 = shl nuw nsw i32 %82, 1
  br label %96

96:                                               ; preds = %108, %94
  %97 = phi i32 [ %109, %108 ], [ %95, %94 ]
  %98 = lshr i32 %97, 5
  %99 = zext nneg i32 %98 to i64
  %100 = getelementptr inbounds nuw i32, ptr %3, i64 %99
  %101 = load i32, ptr %100, align 4, !tbaa !6
  %102 = and i32 %97, 31
  %103 = shl nuw i32 1, %102
  %104 = and i32 %101, %103
  %105 = icmp eq i32 %104, 0
  br i1 %105, label %108, label %106

106:                                              ; preds = %96
  %107 = xor i32 %101, %103
  store i32 %107, ptr %100, align 4, !tbaa !6
  br label %108

108:                                              ; preds = %106, %96
  %109 = add nuw nsw i32 %97, %82
  %110 = icmp samesign ugt i32 %109, 10240000
  br i1 %110, label %111, label %96, !llvm.loop !10

111:                                              ; preds = %108, %91, %80
  %112 = phi i32 [ %81, %80 ], [ %92, %91 ], [ %92, %108 ]
  %113 = add nuw nsw i32 %82, 1
  %114 = icmp eq i32 %113, 10240001
  br i1 %114, label %115, label %80, !llvm.loop !12

115:                                              ; preds = %111
  %116 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef 10240000, i32 noundef %112)
  tail call void @free(ptr noundef nonnull %3) #6
  br label %117

117:                                              ; preds = %2, %115
  %118 = phi i32 [ 0, %115 ], [ 1, %2 ]
  ret i32 %118
}

; Function Attrs: mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite)
declare noalias noundef ptr @malloc(i64 noundef) local_unnamed_addr #1

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #2

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare void @free(ptr allocptr noundef captures(none)) local_unnamed_addr #4

attributes #0 = { nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nofree nounwind willreturn allockind("alloc,uninitialized") allocsize(0) memory(inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind allocsize(0) }
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
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
