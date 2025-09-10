; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/salsa20.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Benchmarks/Misc/salsa20.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@ptr = dso_local local_unnamed_addr global i32 0, align 4
@outbuf = dso_local global [16 x i32] zeroinitializer, align 4
@STATE = dso_local global [16 x i32] zeroinitializer, align 16
@.str = private unnamed_addr constant [15 x i8] c"got:       %x\0A\00", align 1
@.str.1 = private unnamed_addr constant [15 x i8] c"should be: %x\0A\00", align 1

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @salsa20(ptr noundef writeonly captures(none) %0, ptr noundef readonly captures(none) %1) local_unnamed_addr #0 {
  %3 = load i32, ptr %1, align 4, !tbaa !6
  %4 = getelementptr inbounds nuw i8, ptr %1, i64 4
  %5 = load i32, ptr %4, align 4, !tbaa !6
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %7 = load i32, ptr %6, align 4, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 12
  %9 = load i32, ptr %8, align 4, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %11 = load i32, ptr %10, align 4, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 20
  %13 = load i32, ptr %12, align 4, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %15 = load i32, ptr %14, align 4, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 28
  %17 = load i32, ptr %16, align 4, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %19 = load i32, ptr %18, align 4, !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 36
  %21 = load i32, ptr %20, align 4, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %23 = load i32, ptr %22, align 4, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 44
  %25 = load i32, ptr %24, align 4, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %27 = load i32, ptr %26, align 4, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 52
  %29 = load i32, ptr %28, align 4, !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %31 = load i32, ptr %30, align 4, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 60
  %33 = load i32, ptr %32, align 4, !tbaa !6
  br label %82

34:                                               ; preds = %82
  %35 = load i32, ptr %1, align 4, !tbaa !6
  %36 = add i32 %35, %159
  store i32 %36, ptr %0, align 4, !tbaa !6
  %37 = load i32, ptr %4, align 4, !tbaa !6
  %38 = add i32 %37, %150
  %39 = getelementptr inbounds nuw i8, ptr %0, i64 4
  store i32 %38, ptr %39, align 4, !tbaa !6
  %40 = load i32, ptr %6, align 4, !tbaa !6
  %41 = add i32 %40, %153
  %42 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store i32 %41, ptr %42, align 4, !tbaa !6
  %43 = load i32, ptr %8, align 4, !tbaa !6
  %44 = add i32 %43, %156
  %45 = getelementptr inbounds nuw i8, ptr %0, i64 12
  store i32 %44, ptr %45, align 4, !tbaa !6
  %46 = load i32, ptr %10, align 4, !tbaa !6
  %47 = add i32 %46, %168
  %48 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store i32 %47, ptr %48, align 4, !tbaa !6
  %49 = load i32, ptr %12, align 4, !tbaa !6
  %50 = add i32 %49, %171
  %51 = getelementptr inbounds nuw i8, ptr %0, i64 20
  store i32 %50, ptr %51, align 4, !tbaa !6
  %52 = load i32, ptr %14, align 4, !tbaa !6
  %53 = add i32 %52, %162
  %54 = getelementptr inbounds nuw i8, ptr %0, i64 24
  store i32 %53, ptr %54, align 4, !tbaa !6
  %55 = load i32, ptr %16, align 4, !tbaa !6
  %56 = add i32 %55, %165
  %57 = getelementptr inbounds nuw i8, ptr %0, i64 28
  store i32 %56, ptr %57, align 4, !tbaa !6
  %58 = load i32, ptr %18, align 4, !tbaa !6
  %59 = add i32 %58, %177
  %60 = getelementptr inbounds nuw i8, ptr %0, i64 32
  store i32 %59, ptr %60, align 4, !tbaa !6
  %61 = load i32, ptr %20, align 4, !tbaa !6
  %62 = add i32 %61, %180
  %63 = getelementptr inbounds nuw i8, ptr %0, i64 36
  store i32 %62, ptr %63, align 4, !tbaa !6
  %64 = load i32, ptr %22, align 4, !tbaa !6
  %65 = add i32 %64, %183
  %66 = getelementptr inbounds nuw i8, ptr %0, i64 40
  store i32 %65, ptr %66, align 4, !tbaa !6
  %67 = load i32, ptr %24, align 4, !tbaa !6
  %68 = add i32 %67, %174
  %69 = getelementptr inbounds nuw i8, ptr %0, i64 44
  store i32 %68, ptr %69, align 4, !tbaa !6
  %70 = load i32, ptr %26, align 4, !tbaa !6
  %71 = add i32 %70, %186
  %72 = getelementptr inbounds nuw i8, ptr %0, i64 48
  store i32 %71, ptr %72, align 4, !tbaa !6
  %73 = load i32, ptr %28, align 4, !tbaa !6
  %74 = add i32 %73, %189
  %75 = getelementptr inbounds nuw i8, ptr %0, i64 52
  store i32 %74, ptr %75, align 4, !tbaa !6
  %76 = load i32, ptr %30, align 4, !tbaa !6
  %77 = add i32 %76, %192
  %78 = getelementptr inbounds nuw i8, ptr %0, i64 56
  store i32 %77, ptr %78, align 4, !tbaa !6
  %79 = load i32, ptr %32, align 4, !tbaa !6
  %80 = add i32 %79, %195
  %81 = getelementptr inbounds nuw i8, ptr %0, i64 60
  store i32 %80, ptr %81, align 4, !tbaa !6
  ret void

82:                                               ; preds = %2, %82
  %83 = phi i32 [ 20, %2 ], [ %196, %82 ]
  %84 = phi i32 [ %3, %2 ], [ %159, %82 ]
  %85 = phi i32 [ %27, %2 ], [ %186, %82 ]
  %86 = phi i32 [ %11, %2 ], [ %168, %82 ]
  %87 = phi i32 [ %19, %2 ], [ %177, %82 ]
  %88 = phi i32 [ %13, %2 ], [ %171, %82 ]
  %89 = phi i32 [ %5, %2 ], [ %150, %82 ]
  %90 = phi i32 [ %21, %2 ], [ %180, %82 ]
  %91 = phi i32 [ %29, %2 ], [ %189, %82 ]
  %92 = phi i32 [ %23, %2 ], [ %183, %82 ]
  %93 = phi i32 [ %15, %2 ], [ %162, %82 ]
  %94 = phi i32 [ %31, %2 ], [ %192, %82 ]
  %95 = phi i32 [ %7, %2 ], [ %153, %82 ]
  %96 = phi i32 [ %33, %2 ], [ %195, %82 ]
  %97 = phi i32 [ %25, %2 ], [ %174, %82 ]
  %98 = phi i32 [ %9, %2 ], [ %156, %82 ]
  %99 = phi i32 [ %17, %2 ], [ %165, %82 ]
  %100 = add i32 %85, %84
  %101 = tail call i32 @llvm.fshl.i32(i32 %100, i32 %100, i32 7)
  %102 = xor i32 %101, %86
  %103 = add i32 %102, %84
  %104 = tail call i32 @llvm.fshl.i32(i32 %103, i32 %103, i32 9)
  %105 = xor i32 %104, %87
  %106 = add i32 %105, %102
  %107 = tail call i32 @llvm.fshl.i32(i32 %106, i32 %106, i32 13)
  %108 = xor i32 %107, %85
  %109 = add i32 %108, %105
  %110 = tail call i32 @llvm.fshl.i32(i32 %109, i32 %109, i32 18)
  %111 = xor i32 %110, %84
  %112 = add i32 %89, %88
  %113 = tail call i32 @llvm.fshl.i32(i32 %112, i32 %112, i32 7)
  %114 = xor i32 %113, %90
  %115 = add i32 %114, %88
  %116 = tail call i32 @llvm.fshl.i32(i32 %115, i32 %115, i32 9)
  %117 = xor i32 %116, %91
  %118 = add i32 %117, %114
  %119 = tail call i32 @llvm.fshl.i32(i32 %118, i32 %118, i32 13)
  %120 = xor i32 %119, %89
  %121 = add i32 %120, %117
  %122 = tail call i32 @llvm.fshl.i32(i32 %121, i32 %121, i32 18)
  %123 = xor i32 %122, %88
  %124 = add i32 %93, %92
  %125 = tail call i32 @llvm.fshl.i32(i32 %124, i32 %124, i32 7)
  %126 = xor i32 %125, %94
  %127 = add i32 %126, %92
  %128 = tail call i32 @llvm.fshl.i32(i32 %127, i32 %127, i32 9)
  %129 = xor i32 %128, %95
  %130 = add i32 %129, %126
  %131 = tail call i32 @llvm.fshl.i32(i32 %130, i32 %130, i32 13)
  %132 = xor i32 %131, %93
  %133 = add i32 %132, %129
  %134 = tail call i32 @llvm.fshl.i32(i32 %133, i32 %133, i32 18)
  %135 = xor i32 %134, %92
  %136 = add i32 %97, %96
  %137 = tail call i32 @llvm.fshl.i32(i32 %136, i32 %136, i32 7)
  %138 = xor i32 %137, %98
  %139 = add i32 %138, %96
  %140 = tail call i32 @llvm.fshl.i32(i32 %139, i32 %139, i32 9)
  %141 = xor i32 %140, %99
  %142 = add i32 %141, %138
  %143 = tail call i32 @llvm.fshl.i32(i32 %142, i32 %142, i32 13)
  %144 = xor i32 %143, %97
  %145 = add i32 %144, %141
  %146 = tail call i32 @llvm.fshl.i32(i32 %145, i32 %145, i32 18)
  %147 = xor i32 %146, %96
  %148 = add i32 %138, %111
  %149 = tail call i32 @llvm.fshl.i32(i32 %148, i32 %148, i32 7)
  %150 = xor i32 %149, %120
  %151 = add i32 %150, %111
  %152 = tail call i32 @llvm.fshl.i32(i32 %151, i32 %151, i32 9)
  %153 = xor i32 %152, %129
  %154 = add i32 %153, %150
  %155 = tail call i32 @llvm.fshl.i32(i32 %154, i32 %154, i32 13)
  %156 = xor i32 %155, %138
  %157 = add i32 %156, %153
  %158 = tail call i32 @llvm.fshl.i32(i32 %157, i32 %157, i32 18)
  %159 = xor i32 %158, %111
  %160 = add i32 %123, %102
  %161 = tail call i32 @llvm.fshl.i32(i32 %160, i32 %160, i32 7)
  %162 = xor i32 %132, %161
  %163 = add i32 %162, %123
  %164 = tail call i32 @llvm.fshl.i32(i32 %163, i32 %163, i32 9)
  %165 = xor i32 %141, %164
  %166 = add i32 %165, %162
  %167 = tail call i32 @llvm.fshl.i32(i32 %166, i32 %166, i32 13)
  %168 = xor i32 %167, %102
  %169 = add i32 %168, %165
  %170 = tail call i32 @llvm.fshl.i32(i32 %169, i32 %169, i32 18)
  %171 = xor i32 %170, %123
  %172 = add i32 %135, %114
  %173 = tail call i32 @llvm.fshl.i32(i32 %172, i32 %172, i32 7)
  %174 = xor i32 %144, %173
  %175 = add i32 %174, %135
  %176 = tail call i32 @llvm.fshl.i32(i32 %175, i32 %175, i32 9)
  %177 = xor i32 %176, %105
  %178 = add i32 %177, %174
  %179 = tail call i32 @llvm.fshl.i32(i32 %178, i32 %178, i32 13)
  %180 = xor i32 %179, %114
  %181 = add i32 %180, %177
  %182 = tail call i32 @llvm.fshl.i32(i32 %181, i32 %181, i32 18)
  %183 = xor i32 %182, %135
  %184 = add i32 %147, %126
  %185 = tail call i32 @llvm.fshl.i32(i32 %184, i32 %184, i32 7)
  %186 = xor i32 %185, %108
  %187 = add i32 %186, %147
  %188 = tail call i32 @llvm.fshl.i32(i32 %187, i32 %187, i32 9)
  %189 = xor i32 %188, %117
  %190 = add i32 %189, %186
  %191 = tail call i32 @llvm.fshl.i32(i32 %190, i32 %190, i32 13)
  %192 = xor i32 %191, %126
  %193 = add i32 %192, %189
  %194 = tail call i32 @llvm.fshl.i32(i32 %193, i32 %193, i32 18)
  %195 = xor i32 %194, %147
  %196 = add nsw i32 %83, -2
  %197 = icmp eq i32 %196, 0
  br i1 %197, label %34, label %82, !llvm.loop !10
}

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local i32 @salsa() local_unnamed_addr #1 {
  %1 = load i32, ptr @ptr, align 4, !tbaa !6
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %6

3:                                                ; preds = %0
  tail call void @salsa20(ptr noundef nonnull @outbuf, ptr noundef nonnull @STATE)
  %4 = load i32, ptr @STATE, align 4, !tbaa !6
  %5 = add i32 %4, 1
  store i32 %5, ptr @STATE, align 4, !tbaa !6
  br label %6

6:                                                ; preds = %3, %0
  %7 = add i32 %1, 1
  %8 = and i32 %7, 15
  store i32 %8, ptr @ptr, align 4, !tbaa !6
  %9 = zext nneg i32 %8 to i64
  %10 = getelementptr inbounds nuw i32, ptr @outbuf, i64 %9
  %11 = load i32, ptr %10, align 4, !tbaa !6
  ret i32 %11
}

; Function Attrs: nofree nounwind uwtable
define dso_local range(i32 0, 2) i32 @main() local_unnamed_addr #2 {
  store <4 x i32> <i32 856036637, i32 856036637, i32 856036633, i32 856036633>, ptr @STATE, align 16, !tbaa !6
  store <4 x i32> <i32 856036637, i32 856036637, i32 856036625, i32 856036625>, ptr getelementptr inbounds nuw (i8, ptr @STATE, i64 16), align 16, !tbaa !6
  store <4 x i32> <i32 856036637, i32 856036637, i32 856036633, i32 856036633>, ptr getelementptr inbounds nuw (i8, ptr @STATE, i64 32), align 16, !tbaa !6
  store <4 x i32> <i32 856036637, i32 856036637, i32 856036065, i32 856036065>, ptr getelementptr inbounds nuw (i8, ptr @STATE, i64 48), align 16, !tbaa !6
  %1 = load i32, ptr @ptr, align 4, !tbaa !6
  br label %2

2:                                                ; preds = %0, %9
  %3 = phi i32 [ 0, %0 ], [ %13, %9 ]
  %4 = phi i32 [ %1, %0 ], [ %12, %9 ]
  %5 = phi i32 [ 856036637, %0 ], [ %10, %9 ]
  %6 = icmp eq i32 %4, 0
  br i1 %6, label %7, label %9

7:                                                ; preds = %2
  tail call void @salsa20(ptr noundef nonnull @outbuf, ptr noundef nonnull @STATE)
  %8 = add i32 %5, 1
  store i32 %8, ptr @STATE, align 16, !tbaa !6
  br label %9

9:                                                ; preds = %2, %7
  %10 = phi i32 [ %5, %2 ], [ %8, %7 ]
  %11 = add i32 %4, 1
  %12 = and i32 %11, 15
  %13 = add nuw nsw i32 %3, 1
  %14 = icmp eq i32 %13, 537919488
  br i1 %14, label %15, label %2, !llvm.loop !12

15:                                               ; preds = %9
  %16 = and i32 %4, 15
  %17 = icmp eq i32 %16, 15
  br i1 %17, label %18, label %20

18:                                               ; preds = %15
  tail call void @salsa20(ptr noundef nonnull @outbuf, ptr noundef nonnull @STATE)
  %19 = add i32 %10, 1
  store i32 %19, ptr @STATE, align 16, !tbaa !6
  br label %20

20:                                               ; preds = %15, %18
  %21 = add i32 %4, 2
  %22 = and i32 %21, 15
  store i32 %22, ptr @ptr, align 4, !tbaa !6
  %23 = zext nneg i32 %22 to i64
  %24 = getelementptr inbounds nuw i32, ptr @outbuf, i64 %23
  %25 = load i32, ptr %24, align 4, !tbaa !6
  %26 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str, i32 noundef %25)
  %27 = tail call i32 (ptr, ...) @printf(ptr noundef nonnull dereferenceable(1) @.str.1, i32 noundef -162172867)
  %28 = icmp ne i32 %25, -162172867
  %29 = zext i1 %28 to i32
  ret i32 %29
}

; Function Attrs: nofree nounwind
declare noundef i32 @printf(ptr noundef readonly captures(none), ...) local_unnamed_addr #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { nofree nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

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
