; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr28982b.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr28982b.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

%struct.big = type { [65536 x i32] }

@incs = dso_local local_unnamed_addr global [20 x i32] zeroinitializer, align 16
@ptrs = dso_local local_unnamed_addr global [20 x ptr] zeroinitializer, align 8
@results = dso_local local_unnamed_addr global [20 x float] zeroinitializer, align 64
@input = dso_local global [80 x float] zeroinitializer, align 16

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable
define dso_local void @bar(ptr dead_on_return noundef readonly captures(none) %0) local_unnamed_addr #0 {
  %2 = load i32, ptr %0, align 4, !tbaa !6
  %3 = load i32, ptr @incs, align 4, !tbaa !6
  %4 = add nsw i32 %3, %2
  store i32 %4, ptr @incs, align 4, !tbaa !6
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #1 {
  %2 = alloca %struct.big, align 4
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %169, label %4

4:                                                ; preds = %1
  %5 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 152), align 8, !tbaa !10
  %6 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 144), align 8, !tbaa !10
  %7 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 136), align 8, !tbaa !10
  %8 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 128), align 8, !tbaa !10
  %9 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 120), align 8, !tbaa !10
  %10 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 112), align 8, !tbaa !10
  %11 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 104), align 8, !tbaa !10
  %12 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 96), align 8, !tbaa !10
  %13 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 88), align 8, !tbaa !10
  %14 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 80), align 8, !tbaa !10
  %15 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 72), align 8, !tbaa !10
  %16 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 64), align 8, !tbaa !10
  %17 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 56), align 8, !tbaa !10
  %18 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 48), align 8, !tbaa !10
  %19 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 40), align 8, !tbaa !10
  %20 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 32), align 8, !tbaa !10
  %21 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 24), align 8, !tbaa !10
  %22 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 16), align 8, !tbaa !10
  %23 = load ptr, ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 8), align 8, !tbaa !10
  %24 = load ptr, ptr @ptrs, align 8, !tbaa !10
  %25 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 76), align 4, !tbaa !6
  %26 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 72), align 4, !tbaa !6
  %27 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 68), align 4, !tbaa !6
  %28 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 64), align 4, !tbaa !6
  %29 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 60), align 4, !tbaa !6
  %30 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 56), align 4, !tbaa !6
  %31 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 52), align 4, !tbaa !6
  %32 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 48), align 4, !tbaa !6
  %33 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 44), align 4, !tbaa !6
  %34 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 40), align 4, !tbaa !6
  %35 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 36), align 4, !tbaa !6
  %36 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 32), align 4, !tbaa !6
  %37 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 28), align 4, !tbaa !6
  %38 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 24), align 4, !tbaa !6
  %39 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 20), align 4, !tbaa !6
  %40 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 16), align 4, !tbaa !6
  %41 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 12), align 4, !tbaa !6
  %42 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 8), align 4, !tbaa !6
  %43 = load i32, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 4), align 4, !tbaa !6
  %44 = load i32, ptr @incs, align 4, !tbaa !6
  %45 = sext i32 %44 to i64
  %46 = sext i32 %43 to i64
  %47 = sext i32 %42 to i64
  %48 = sext i32 %41 to i64
  %49 = sext i32 %40 to i64
  %50 = sext i32 %39 to i64
  %51 = sext i32 %38 to i64
  %52 = sext i32 %37 to i64
  %53 = sext i32 %36 to i64
  %54 = sext i32 %35 to i64
  %55 = sext i32 %34 to i64
  %56 = sext i32 %33 to i64
  %57 = sext i32 %32 to i64
  %58 = sext i32 %31 to i64
  %59 = sext i32 %30 to i64
  %60 = sext i32 %29 to i64
  %61 = sext i32 %28 to i64
  %62 = sext i32 %27 to i64
  %63 = sext i32 %26 to i64
  %64 = sext i32 %25 to i64
  br label %65

65:                                               ; preds = %4, %65
  %66 = phi float [ 0.000000e+00, %4 ], [ %166, %65 ]
  %67 = phi ptr [ %5, %4 ], [ %167, %65 ]
  %68 = phi float [ 0.000000e+00, %4 ], [ %163, %65 ]
  %69 = phi ptr [ %6, %4 ], [ %164, %65 ]
  %70 = phi float [ 0.000000e+00, %4 ], [ %160, %65 ]
  %71 = phi ptr [ %7, %4 ], [ %161, %65 ]
  %72 = phi float [ 0.000000e+00, %4 ], [ %157, %65 ]
  %73 = phi ptr [ %8, %4 ], [ %158, %65 ]
  %74 = phi float [ 0.000000e+00, %4 ], [ %154, %65 ]
  %75 = phi ptr [ %9, %4 ], [ %155, %65 ]
  %76 = phi float [ 0.000000e+00, %4 ], [ %151, %65 ]
  %77 = phi ptr [ %10, %4 ], [ %152, %65 ]
  %78 = phi float [ 0.000000e+00, %4 ], [ %148, %65 ]
  %79 = phi ptr [ %11, %4 ], [ %149, %65 ]
  %80 = phi float [ 0.000000e+00, %4 ], [ %145, %65 ]
  %81 = phi ptr [ %12, %4 ], [ %146, %65 ]
  %82 = phi float [ 0.000000e+00, %4 ], [ %142, %65 ]
  %83 = phi ptr [ %13, %4 ], [ %143, %65 ]
  %84 = phi float [ 0.000000e+00, %4 ], [ %139, %65 ]
  %85 = phi ptr [ %14, %4 ], [ %140, %65 ]
  %86 = phi float [ 0.000000e+00, %4 ], [ %136, %65 ]
  %87 = phi ptr [ %15, %4 ], [ %137, %65 ]
  %88 = phi float [ 0.000000e+00, %4 ], [ %133, %65 ]
  %89 = phi ptr [ %16, %4 ], [ %134, %65 ]
  %90 = phi float [ 0.000000e+00, %4 ], [ %130, %65 ]
  %91 = phi ptr [ %17, %4 ], [ %131, %65 ]
  %92 = phi float [ 0.000000e+00, %4 ], [ %127, %65 ]
  %93 = phi ptr [ %18, %4 ], [ %128, %65 ]
  %94 = phi float [ 0.000000e+00, %4 ], [ %124, %65 ]
  %95 = phi ptr [ %19, %4 ], [ %125, %65 ]
  %96 = phi float [ 0.000000e+00, %4 ], [ %121, %65 ]
  %97 = phi ptr [ %20, %4 ], [ %122, %65 ]
  %98 = phi float [ 0.000000e+00, %4 ], [ %118, %65 ]
  %99 = phi ptr [ %21, %4 ], [ %119, %65 ]
  %100 = phi float [ 0.000000e+00, %4 ], [ %115, %65 ]
  %101 = phi ptr [ %22, %4 ], [ %116, %65 ]
  %102 = phi float [ 0.000000e+00, %4 ], [ %112, %65 ]
  %103 = phi ptr [ %23, %4 ], [ %113, %65 ]
  %104 = phi float [ 0.000000e+00, %4 ], [ %109, %65 ]
  %105 = phi ptr [ %24, %4 ], [ %110, %65 ]
  %106 = phi i32 [ %0, %4 ], [ %107, %65 ]
  %107 = add nsw i32 %106, -1
  %108 = load float, ptr %105, align 4, !tbaa !13
  %109 = fadd float %104, %108
  %110 = getelementptr inbounds float, ptr %105, i64 %45
  %111 = load float, ptr %103, align 4, !tbaa !13
  %112 = fadd float %102, %111
  %113 = getelementptr inbounds float, ptr %103, i64 %46
  %114 = load float, ptr %101, align 4, !tbaa !13
  %115 = fadd float %100, %114
  %116 = getelementptr inbounds float, ptr %101, i64 %47
  %117 = load float, ptr %99, align 4, !tbaa !13
  %118 = fadd float %98, %117
  %119 = getelementptr inbounds float, ptr %99, i64 %48
  %120 = load float, ptr %97, align 4, !tbaa !13
  %121 = fadd float %96, %120
  %122 = getelementptr inbounds float, ptr %97, i64 %49
  %123 = load float, ptr %95, align 4, !tbaa !13
  %124 = fadd float %94, %123
  %125 = getelementptr inbounds float, ptr %95, i64 %50
  %126 = load float, ptr %93, align 4, !tbaa !13
  %127 = fadd float %92, %126
  %128 = getelementptr inbounds float, ptr %93, i64 %51
  %129 = load float, ptr %91, align 4, !tbaa !13
  %130 = fadd float %90, %129
  %131 = getelementptr inbounds float, ptr %91, i64 %52
  %132 = load float, ptr %89, align 4, !tbaa !13
  %133 = fadd float %88, %132
  %134 = getelementptr inbounds float, ptr %89, i64 %53
  %135 = load float, ptr %87, align 4, !tbaa !13
  %136 = fadd float %86, %135
  %137 = getelementptr inbounds float, ptr %87, i64 %54
  %138 = load float, ptr %85, align 4, !tbaa !13
  %139 = fadd float %84, %138
  %140 = getelementptr inbounds float, ptr %85, i64 %55
  %141 = load float, ptr %83, align 4, !tbaa !13
  %142 = fadd float %82, %141
  %143 = getelementptr inbounds float, ptr %83, i64 %56
  %144 = load float, ptr %81, align 4, !tbaa !13
  %145 = fadd float %80, %144
  %146 = getelementptr inbounds float, ptr %81, i64 %57
  %147 = load float, ptr %79, align 4, !tbaa !13
  %148 = fadd float %78, %147
  %149 = getelementptr inbounds float, ptr %79, i64 %58
  %150 = load float, ptr %77, align 4, !tbaa !13
  %151 = fadd float %76, %150
  %152 = getelementptr inbounds float, ptr %77, i64 %59
  %153 = load float, ptr %75, align 4, !tbaa !13
  %154 = fadd float %74, %153
  %155 = getelementptr inbounds float, ptr %75, i64 %60
  %156 = load float, ptr %73, align 4, !tbaa !13
  %157 = fadd float %72, %156
  %158 = getelementptr inbounds float, ptr %73, i64 %61
  %159 = load float, ptr %71, align 4, !tbaa !13
  %160 = fadd float %70, %159
  %161 = getelementptr inbounds float, ptr %71, i64 %62
  %162 = load float, ptr %69, align 4, !tbaa !13
  %163 = fadd float %68, %162
  %164 = getelementptr inbounds float, ptr %69, i64 %63
  %165 = load float, ptr %67, align 4, !tbaa !13
  %166 = fadd float %66, %165
  %167 = getelementptr inbounds float, ptr %67, i64 %64
  %168 = icmp eq i32 %107, 0
  br i1 %168, label %169, label %65, !llvm.loop !15

169:                                              ; preds = %65, %1
  %170 = phi float [ 0.000000e+00, %1 ], [ %109, %65 ]
  %171 = phi float [ 0.000000e+00, %1 ], [ %112, %65 ]
  %172 = phi float [ 0.000000e+00, %1 ], [ %115, %65 ]
  %173 = phi float [ 0.000000e+00, %1 ], [ %118, %65 ]
  %174 = phi float [ 0.000000e+00, %1 ], [ %121, %65 ]
  %175 = phi float [ 0.000000e+00, %1 ], [ %124, %65 ]
  %176 = phi float [ 0.000000e+00, %1 ], [ %127, %65 ]
  %177 = phi float [ 0.000000e+00, %1 ], [ %130, %65 ]
  %178 = phi float [ 0.000000e+00, %1 ], [ %133, %65 ]
  %179 = phi float [ 0.000000e+00, %1 ], [ %136, %65 ]
  %180 = phi float [ 0.000000e+00, %1 ], [ %139, %65 ]
  %181 = phi float [ 0.000000e+00, %1 ], [ %142, %65 ]
  %182 = phi float [ 0.000000e+00, %1 ], [ %145, %65 ]
  %183 = phi float [ 0.000000e+00, %1 ], [ %148, %65 ]
  %184 = phi float [ 0.000000e+00, %1 ], [ %151, %65 ]
  %185 = phi float [ 0.000000e+00, %1 ], [ %154, %65 ]
  %186 = phi float [ 0.000000e+00, %1 ], [ %157, %65 ]
  %187 = phi float [ 0.000000e+00, %1 ], [ %160, %65 ]
  %188 = phi float [ 0.000000e+00, %1 ], [ %163, %65 ]
  %189 = phi float [ 0.000000e+00, %1 ], [ %166, %65 ]
  store float %170, ptr @results, align 4, !tbaa !13
  store float %171, ptr getelementptr inbounds nuw (i8, ptr @results, i64 4), align 4, !tbaa !13
  store float %172, ptr getelementptr inbounds nuw (i8, ptr @results, i64 8), align 4, !tbaa !13
  store float %173, ptr getelementptr inbounds nuw (i8, ptr @results, i64 12), align 4, !tbaa !13
  store float %174, ptr getelementptr inbounds nuw (i8, ptr @results, i64 16), align 4, !tbaa !13
  store float %175, ptr getelementptr inbounds nuw (i8, ptr @results, i64 20), align 4, !tbaa !13
  store float %176, ptr getelementptr inbounds nuw (i8, ptr @results, i64 24), align 4, !tbaa !13
  store float %177, ptr getelementptr inbounds nuw (i8, ptr @results, i64 28), align 4, !tbaa !13
  store float %178, ptr getelementptr inbounds nuw (i8, ptr @results, i64 32), align 4, !tbaa !13
  store float %179, ptr getelementptr inbounds nuw (i8, ptr @results, i64 36), align 4, !tbaa !13
  store float %180, ptr getelementptr inbounds nuw (i8, ptr @results, i64 40), align 4, !tbaa !13
  store float %181, ptr getelementptr inbounds nuw (i8, ptr @results, i64 44), align 4, !tbaa !13
  store float %182, ptr getelementptr inbounds nuw (i8, ptr @results, i64 48), align 4, !tbaa !13
  store float %183, ptr getelementptr inbounds nuw (i8, ptr @results, i64 52), align 4, !tbaa !13
  store float %184, ptr getelementptr inbounds nuw (i8, ptr @results, i64 56), align 4, !tbaa !13
  store float %185, ptr getelementptr inbounds nuw (i8, ptr @results, i64 60), align 4, !tbaa !13
  store float %186, ptr getelementptr inbounds nuw (i8, ptr @results, i64 64), align 4, !tbaa !13
  store float %187, ptr getelementptr inbounds nuw (i8, ptr @results, i64 68), align 4, !tbaa !13
  store float %188, ptr getelementptr inbounds nuw (i8, ptr @results, i64 72), align 4, !tbaa !13
  store float %189, ptr getelementptr inbounds nuw (i8, ptr @results, i64 76), align 4, !tbaa !13
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(262144) %2, i8 0, i64 262144, i1 false)
  call void @bar(ptr dead_on_return noundef nonnull %2)
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #4 {
  store ptr @input, ptr @ptrs, align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 4), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 8), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 8), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 16), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 12), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 24), align 8, !tbaa !10
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, ptr @incs, align 16, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 16), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 32), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 20), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 40), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 24), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 48), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 28), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 56), align 8, !tbaa !10
  store <4 x i32> <i32 4, i32 5, i32 6, i32 7>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 16), align 16, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 32), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 64), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 36), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 72), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 40), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 80), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 44), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 88), align 8, !tbaa !10
  store <4 x i32> <i32 8, i32 9, i32 10, i32 11>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 32), align 16, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 48), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 96), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 52), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 104), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 56), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 112), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 60), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 120), align 8, !tbaa !10
  store <4 x i32> <i32 12, i32 13, i32 14, i32 15>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 48), align 16, !tbaa !6
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 64), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 128), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 68), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 136), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 72), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 144), align 8, !tbaa !10
  store ptr getelementptr inbounds nuw (i8, ptr @input, i64 76), ptr getelementptr inbounds nuw (i8, ptr @ptrs, i64 152), align 8, !tbaa !10
  store <4 x i32> <i32 16, i32 17, i32 18, i32 19>, ptr getelementptr inbounds nuw (i8, ptr @incs, i64 64), align 16, !tbaa !6
  store <4 x float> <float 0.000000e+00, float 1.000000e+00, float 2.000000e+00, float 3.000000e+00>, ptr @input, align 16, !tbaa !13
  store <4 x float> <float 4.000000e+00, float 5.000000e+00, float 6.000000e+00, float 7.000000e+00>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 16), align 16, !tbaa !13
  store <4 x float> <float 8.000000e+00, float 9.000000e+00, float 1.000000e+01, float 1.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 32), align 16, !tbaa !13
  store <4 x float> <float 1.200000e+01, float 1.300000e+01, float 1.400000e+01, float 1.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 48), align 16, !tbaa !13
  store <4 x float> <float 1.600000e+01, float 1.700000e+01, float 1.800000e+01, float 1.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 64), align 16, !tbaa !13
  store <4 x float> <float 2.000000e+01, float 2.100000e+01, float 2.200000e+01, float 2.300000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 80), align 16, !tbaa !13
  store <4 x float> <float 2.400000e+01, float 2.500000e+01, float 2.600000e+01, float 2.700000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 96), align 16, !tbaa !13
  store <4 x float> <float 2.800000e+01, float 2.900000e+01, float 3.000000e+01, float 3.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 112), align 16, !tbaa !13
  store <4 x float> <float 3.200000e+01, float 3.300000e+01, float 3.400000e+01, float 3.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 128), align 16, !tbaa !13
  store <4 x float> <float 3.600000e+01, float 3.700000e+01, float 3.800000e+01, float 3.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 144), align 16, !tbaa !13
  store <4 x float> <float 4.000000e+01, float 4.100000e+01, float 4.200000e+01, float 4.300000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 160), align 16, !tbaa !13
  store <4 x float> <float 4.400000e+01, float 4.500000e+01, float 4.600000e+01, float 4.700000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 176), align 16, !tbaa !13
  store <4 x float> <float 4.800000e+01, float 4.900000e+01, float 5.000000e+01, float 5.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 192), align 16, !tbaa !13
  store <4 x float> <float 5.200000e+01, float 5.300000e+01, float 5.400000e+01, float 5.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 208), align 16, !tbaa !13
  store <4 x float> <float 5.600000e+01, float 5.700000e+01, float 5.800000e+01, float 5.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 224), align 16, !tbaa !13
  store <4 x float> <float 6.000000e+01, float 6.100000e+01, float 6.200000e+01, float 6.300000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 240), align 16, !tbaa !13
  store <4 x float> <float 6.400000e+01, float 6.500000e+01, float 6.600000e+01, float 6.700000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 256), align 16, !tbaa !13
  store <4 x float> <float 6.800000e+01, float 6.900000e+01, float 7.000000e+01, float 7.100000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 272), align 16, !tbaa !13
  store <4 x float> <float 7.200000e+01, float 7.300000e+01, float 7.400000e+01, float 7.500000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 288), align 16, !tbaa !13
  store <4 x float> <float 7.600000e+01, float 7.700000e+01, float 7.800000e+01, float 7.900000e+01>, ptr getelementptr inbounds nuw (i8, ptr @input, i64 304), align 16, !tbaa !13
  tail call void @foo(i32 noundef 4)
  %1 = load <16 x float>, ptr @results, align 64
  %2 = freeze <16 x float> %1
  %3 = fcmp une <16 x float> %2, <float 0.000000e+00, float 1.000000e+01, float 2.000000e+01, float 3.000000e+01, float 4.000000e+01, float 5.000000e+01, float 6.000000e+01, float 7.000000e+01, float 8.000000e+01, float 9.000000e+01, float 1.000000e+02, float 1.100000e+02, float 1.200000e+02, float 1.300000e+02, float 1.400000e+02, float 1.500000e+02>
  %4 = load <4 x float>, ptr getelementptr inbounds nuw (i8, ptr @results, i64 64), align 64
  %5 = freeze <4 x float> %4
  %6 = fcmp une <4 x float> %5, <float 1.600000e+02, float 1.700000e+02, float 1.800000e+02, float 1.900000e+02>
  %7 = shufflevector <16 x i1> %3, <16 x i1> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  %8 = or <4 x i1> %7, %6
  %9 = shufflevector <4 x i1> %8, <4 x i1> poison, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison, i32 poison>
  %10 = shufflevector <16 x i1> %9, <16 x i1> %3, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  %11 = bitcast <16 x i1> %10 to i16
  %12 = icmp ne i16 %11, 0
  %13 = zext i1 %12 to i32
  ret i32 %13
}

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(readwrite, argmem: read, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noinline norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #4 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #5 = { nounwind }

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
!11 = !{!"p1 float", !12, i64 0}
!12 = !{!"any pointer", !8, i64 0}
!13 = !{!14, !14, i64 0}
!14 = !{!"float", !8, i64 0}
!15 = distinct !{!15, !16}
!16 = !{!"llvm.loop.mustprogress"}
