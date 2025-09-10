; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/postmod-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/postmod-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@counter0 = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@counter1 = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@counter2 = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@counter3 = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@counter4 = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@counter5 = dso_local local_unnamed_addr global float 0.000000e+00, align 4
@stop = dso_local global i32 1, align 4
@array0 = dso_local local_unnamed_addr global [16 x float] zeroinitializer, align 4
@array1 = dso_local local_unnamed_addr global [16 x float] zeroinitializer, align 4
@array2 = dso_local local_unnamed_addr global [16 x float] zeroinitializer, align 4
@array3 = dso_local local_unnamed_addr global [16 x float] zeroinitializer, align 4
@array4 = dso_local local_unnamed_addr global [16 x float] zeroinitializer, align 4
@array5 = dso_local local_unnamed_addr global [16 x float] zeroinitializer, align 4
@vol = dso_local global i32 0, align 4

; Function Attrs: nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable
define dso_local void @foo(i32 noundef %0) local_unnamed_addr #0 {
  %2 = sext i32 %0 to i64
  %3 = getelementptr inbounds float, ptr @array0, i64 %2
  %4 = getelementptr inbounds float, ptr @array1, i64 %2
  %5 = getelementptr inbounds float, ptr @array2, i64 %2
  %6 = getelementptr inbounds float, ptr @array3, i64 %2
  %7 = getelementptr inbounds float, ptr @array4, i64 %2
  %8 = getelementptr inbounds float, ptr @array5, i64 %2
  %9 = load float, ptr @counter0, align 4, !tbaa !6
  %10 = load float, ptr @counter1, align 4, !tbaa !6
  %11 = load float, ptr @counter2, align 4, !tbaa !6
  %12 = load float, ptr @counter3, align 4, !tbaa !6
  %13 = load float, ptr @counter4, align 4, !tbaa !6
  %14 = load float, ptr @counter5, align 4, !tbaa !6
  br label %15

15:                                               ; preds = %15, %1
  %16 = phi float [ %14, %1 ], [ %63, %15 ]
  %17 = phi float [ %13, %1 ], [ %60, %15 ]
  %18 = phi float [ %12, %1 ], [ %57, %15 ]
  %19 = phi float [ %11, %1 ], [ %54, %15 ]
  %20 = phi float [ %10, %1 ], [ %51, %15 ]
  %21 = phi float [ %9, %1 ], [ %48, %15 ]
  %22 = phi ptr [ %8, %1 ], [ %45, %15 ]
  %23 = phi ptr [ %7, %1 ], [ %42, %15 ]
  %24 = phi ptr [ %6, %1 ], [ %39, %15 ]
  %25 = phi ptr [ %5, %1 ], [ %36, %15 ]
  %26 = phi ptr [ %4, %1 ], [ %33, %15 ]
  %27 = phi ptr [ %3, %1 ], [ %30, %15 ]
  %28 = load float, ptr %27, align 4, !tbaa !6
  %29 = fadd float %28, %21
  %30 = getelementptr inbounds nuw i8, ptr %27, i64 12
  %31 = load float, ptr %26, align 4, !tbaa !6
  %32 = fadd float %31, %20
  %33 = getelementptr inbounds nuw i8, ptr %26, i64 12
  %34 = load float, ptr %25, align 4, !tbaa !6
  %35 = fadd float %34, %19
  %36 = getelementptr inbounds nuw i8, ptr %25, i64 12
  %37 = load float, ptr %24, align 4, !tbaa !6
  %38 = fadd float %37, %18
  %39 = getelementptr inbounds nuw i8, ptr %24, i64 12
  %40 = load float, ptr %23, align 4, !tbaa !6
  %41 = fadd float %40, %17
  %42 = getelementptr inbounds nuw i8, ptr %23, i64 12
  %43 = load float, ptr %22, align 4, !tbaa !6
  %44 = fadd float %43, %16
  %45 = getelementptr inbounds nuw i8, ptr %22, i64 12
  %46 = getelementptr inbounds float, ptr %30, i64 %2
  %47 = load float, ptr %46, align 4, !tbaa !6
  %48 = fadd float %29, %47
  store float %48, ptr @counter0, align 4, !tbaa !6
  %49 = getelementptr inbounds float, ptr %33, i64 %2
  %50 = load float, ptr %49, align 4, !tbaa !6
  %51 = fadd float %32, %50
  store float %51, ptr @counter1, align 4, !tbaa !6
  %52 = getelementptr inbounds float, ptr %36, i64 %2
  %53 = load float, ptr %52, align 4, !tbaa !6
  %54 = fadd float %35, %53
  store float %54, ptr @counter2, align 4, !tbaa !6
  %55 = getelementptr inbounds float, ptr %39, i64 %2
  %56 = load float, ptr %55, align 4, !tbaa !6
  %57 = fadd float %38, %56
  store float %57, ptr @counter3, align 4, !tbaa !6
  %58 = getelementptr inbounds float, ptr %42, i64 %2
  %59 = load float, ptr %58, align 4, !tbaa !6
  %60 = fadd float %41, %59
  store float %60, ptr @counter4, align 4, !tbaa !6
  %61 = getelementptr inbounds float, ptr %45, i64 %2
  %62 = load float, ptr %61, align 4, !tbaa !6
  %63 = fadd float %44, %62
  store float %63, ptr @counter5, align 4, !tbaa !6
  %64 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %65 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %66 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %67 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %68 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %69 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %70 = add nsw i32 %69, %64
  store volatile i32 %70, ptr @vol, align 4, !tbaa !10
  %71 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %72 = add nsw i32 %71, %65
  store volatile i32 %72, ptr @vol, align 4, !tbaa !10
  %73 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %74 = add nsw i32 %73, %66
  store volatile i32 %74, ptr @vol, align 4, !tbaa !10
  %75 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %76 = add nsw i32 %75, %67
  store volatile i32 %76, ptr @vol, align 4, !tbaa !10
  %77 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %78 = add nsw i32 %77, %68
  store volatile i32 %78, ptr @vol, align 4, !tbaa !10
  %79 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %80 = add nsw i32 %79, %64
  store volatile i32 %80, ptr @vol, align 4, !tbaa !10
  %81 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %82 = add nsw i32 %81, %65
  store volatile i32 %82, ptr @vol, align 4, !tbaa !10
  %83 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %84 = add nsw i32 %83, %66
  store volatile i32 %84, ptr @vol, align 4, !tbaa !10
  %85 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %86 = add nsw i32 %85, %67
  store volatile i32 %86, ptr @vol, align 4, !tbaa !10
  %87 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %88 = add nsw i32 %87, %68
  store volatile i32 %88, ptr @vol, align 4, !tbaa !10
  %89 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %90 = add nsw i32 %89, %64
  store volatile i32 %90, ptr @vol, align 4, !tbaa !10
  %91 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %92 = add nsw i32 %91, %65
  store volatile i32 %92, ptr @vol, align 4, !tbaa !10
  %93 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %94 = add nsw i32 %93, %66
  store volatile i32 %94, ptr @vol, align 4, !tbaa !10
  %95 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %96 = add nsw i32 %95, %67
  store volatile i32 %96, ptr @vol, align 4, !tbaa !10
  %97 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %98 = add nsw i32 %97, %68
  store volatile i32 %98, ptr @vol, align 4, !tbaa !10
  %99 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %100 = add nsw i32 %99, %64
  store volatile i32 %100, ptr @vol, align 4, !tbaa !10
  %101 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %102 = add nsw i32 %101, %65
  store volatile i32 %102, ptr @vol, align 4, !tbaa !10
  %103 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %104 = add nsw i32 %103, %66
  store volatile i32 %104, ptr @vol, align 4, !tbaa !10
  %105 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %106 = add nsw i32 %105, %67
  store volatile i32 %106, ptr @vol, align 4, !tbaa !10
  %107 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %108 = add nsw i32 %107, %68
  store volatile i32 %108, ptr @vol, align 4, !tbaa !10
  %109 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %110 = add nsw i32 %109, %64
  store volatile i32 %110, ptr @vol, align 4, !tbaa !10
  %111 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %112 = add nsw i32 %111, %65
  store volatile i32 %112, ptr @vol, align 4, !tbaa !10
  %113 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %114 = add nsw i32 %113, %66
  store volatile i32 %114, ptr @vol, align 4, !tbaa !10
  %115 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %116 = add nsw i32 %115, %67
  store volatile i32 %116, ptr @vol, align 4, !tbaa !10
  %117 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %118 = add nsw i32 %117, %68
  store volatile i32 %118, ptr @vol, align 4, !tbaa !10
  %119 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %120 = add nsw i32 %119, %64
  store volatile i32 %120, ptr @vol, align 4, !tbaa !10
  %121 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %122 = add nsw i32 %121, %65
  store volatile i32 %122, ptr @vol, align 4, !tbaa !10
  %123 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %124 = add nsw i32 %123, %66
  store volatile i32 %124, ptr @vol, align 4, !tbaa !10
  %125 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %126 = add nsw i32 %125, %67
  store volatile i32 %126, ptr @vol, align 4, !tbaa !10
  %127 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %128 = add nsw i32 %127, %68
  store volatile i32 %128, ptr @vol, align 4, !tbaa !10
  %129 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %130 = add nsw i32 %129, %64
  store volatile i32 %130, ptr @vol, align 4, !tbaa !10
  %131 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %132 = add nsw i32 %131, %65
  store volatile i32 %132, ptr @vol, align 4, !tbaa !10
  %133 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %134 = add nsw i32 %133, %66
  store volatile i32 %134, ptr @vol, align 4, !tbaa !10
  %135 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %136 = add nsw i32 %135, %67
  store volatile i32 %136, ptr @vol, align 4, !tbaa !10
  %137 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %138 = add nsw i32 %137, %68
  store volatile i32 %138, ptr @vol, align 4, !tbaa !10
  %139 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %140 = add nsw i32 %139, %64
  store volatile i32 %140, ptr @vol, align 4, !tbaa !10
  %141 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %142 = add nsw i32 %141, %65
  store volatile i32 %142, ptr @vol, align 4, !tbaa !10
  %143 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %144 = add nsw i32 %143, %66
  store volatile i32 %144, ptr @vol, align 4, !tbaa !10
  %145 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %146 = add nsw i32 %145, %67
  store volatile i32 %146, ptr @vol, align 4, !tbaa !10
  %147 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %148 = add nsw i32 %147, %68
  store volatile i32 %148, ptr @vol, align 4, !tbaa !10
  %149 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %150 = add nsw i32 %149, %64
  store volatile i32 %150, ptr @vol, align 4, !tbaa !10
  %151 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %152 = add nsw i32 %151, %65
  store volatile i32 %152, ptr @vol, align 4, !tbaa !10
  %153 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %154 = add nsw i32 %153, %66
  store volatile i32 %154, ptr @vol, align 4, !tbaa !10
  %155 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %156 = add nsw i32 %155, %67
  store volatile i32 %156, ptr @vol, align 4, !tbaa !10
  %157 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %158 = add nsw i32 %157, %68
  store volatile i32 %158, ptr @vol, align 4, !tbaa !10
  %159 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %160 = add nsw i32 %159, %64
  store volatile i32 %160, ptr @vol, align 4, !tbaa !10
  %161 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %162 = add nsw i32 %161, %65
  store volatile i32 %162, ptr @vol, align 4, !tbaa !10
  %163 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %164 = add nsw i32 %163, %66
  store volatile i32 %164, ptr @vol, align 4, !tbaa !10
  %165 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %166 = add nsw i32 %165, %67
  store volatile i32 %166, ptr @vol, align 4, !tbaa !10
  %167 = load volatile i32, ptr @vol, align 4, !tbaa !10
  %168 = add nsw i32 %167, %68
  store volatile i32 %168, ptr @vol, align 4, !tbaa !10
  %169 = load volatile i32, ptr @stop, align 4, !tbaa !10
  %170 = icmp eq i32 %169, 0
  br i1 %170, label %15, label %171, !llvm.loop !12

171:                                              ; preds = %15
  ret void
}

; Function Attrs: nofree norecurse nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #1 {
  store float 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array0, i64 4), align 4, !tbaa !6
  store float 2.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array0, i64 20), align 4, !tbaa !6
  store float 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array1, i64 4), align 4, !tbaa !6
  store float 2.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array1, i64 20), align 4, !tbaa !6
  store float 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 4), align 4, !tbaa !6
  store float 2.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array2, i64 20), align 4, !tbaa !6
  store float 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array3, i64 4), align 4, !tbaa !6
  store float 2.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array3, i64 20), align 4, !tbaa !6
  store float 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array4, i64 4), align 4, !tbaa !6
  store float 2.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array4, i64 20), align 4, !tbaa !6
  store float 1.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array5, i64 4), align 4, !tbaa !6
  store float 2.000000e+00, ptr getelementptr inbounds nuw (i8, ptr @array5, i64 20), align 4, !tbaa !6
  tail call void @foo(i32 noundef 1)
  %1 = load float, ptr @counter0, align 4, !tbaa !6
  %2 = fcmp une float %1, 3.000000e+00
  %3 = load float, ptr @counter1, align 4, !tbaa !6
  %4 = fcmp une float %3, 3.000000e+00
  %5 = or i1 %2, %4
  %6 = load float, ptr @counter2, align 4, !tbaa !6
  %7 = fcmp une float %6, 3.000000e+00
  %8 = or i1 %5, %7
  %9 = load float, ptr @counter3, align 4, !tbaa !6
  %10 = fcmp une float %9, 3.000000e+00
  %11 = or i1 %8, %10
  %12 = load float, ptr @counter4, align 4, !tbaa !6
  %13 = fcmp une float %12, 3.000000e+00
  %14 = or i1 %11, %13
  %15 = load float, ptr @counter5, align 4, !tbaa !6
  %16 = fcmp une float %15, 3.000000e+00
  %17 = or i1 %14, %16
  %18 = zext i1 %17 to i32
  ret i32 %18
}

attributes #0 = { nofree noinline norecurse nounwind memory(readwrite, argmem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"float", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
