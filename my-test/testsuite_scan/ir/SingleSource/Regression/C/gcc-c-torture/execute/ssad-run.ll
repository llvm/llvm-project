; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ssad-run.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/ssad-run.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable
define dso_local void @bar(ptr noundef readonly captures(none) %0, ptr noundef readonly captures(none) %1, i32 noundef %2, ptr noundef writeonly captures(none) %3) local_unnamed_addr #0 {
  %5 = sext i32 %2 to i64
  %6 = load <16 x i8>, ptr %0, align 1, !tbaa !6
  %7 = sext <16 x i8> %6 to <16 x i16>
  %8 = load <16 x i8>, ptr %1, align 1, !tbaa !6
  %9 = sext <16 x i8> %8 to <16 x i16>
  %10 = sub nsw <16 x i16> %7, %9
  %11 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %10, i1 false)
  %12 = zext <16 x i16> %11 to <16 x i32>
  %13 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %12)
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %15 = getelementptr inbounds i8, ptr %1, i64 %5
  %16 = load <16 x i8>, ptr %14, align 1, !tbaa !6
  %17 = sext <16 x i8> %16 to <16 x i16>
  %18 = load <16 x i8>, ptr %15, align 1, !tbaa !6
  %19 = sext <16 x i8> %18 to <16 x i16>
  %20 = sub nsw <16 x i16> %17, %19
  %21 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %20, i1 false)
  %22 = zext <16 x i16> %21 to <16 x i32>
  %23 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %22)
  %24 = add i32 %23, %13
  %25 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %26 = getelementptr inbounds i8, ptr %15, i64 %5
  %27 = load <16 x i8>, ptr %25, align 1, !tbaa !6
  %28 = sext <16 x i8> %27 to <16 x i16>
  %29 = load <16 x i8>, ptr %26, align 1, !tbaa !6
  %30 = sext <16 x i8> %29 to <16 x i16>
  %31 = sub nsw <16 x i16> %28, %30
  %32 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %31, i1 false)
  %33 = zext <16 x i16> %32 to <16 x i32>
  %34 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %33)
  %35 = add i32 %34, %24
  %36 = getelementptr inbounds nuw i8, ptr %0, i64 48
  %37 = getelementptr inbounds i8, ptr %26, i64 %5
  %38 = load <16 x i8>, ptr %36, align 1, !tbaa !6
  %39 = sext <16 x i8> %38 to <16 x i16>
  %40 = load <16 x i8>, ptr %37, align 1, !tbaa !6
  %41 = sext <16 x i8> %40 to <16 x i16>
  %42 = sub nsw <16 x i16> %39, %41
  %43 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %42, i1 false)
  %44 = zext <16 x i16> %43 to <16 x i32>
  %45 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %44)
  %46 = add i32 %45, %35
  %47 = getelementptr inbounds nuw i8, ptr %0, i64 64
  %48 = getelementptr inbounds i8, ptr %37, i64 %5
  %49 = load <16 x i8>, ptr %47, align 1, !tbaa !6
  %50 = sext <16 x i8> %49 to <16 x i16>
  %51 = load <16 x i8>, ptr %48, align 1, !tbaa !6
  %52 = sext <16 x i8> %51 to <16 x i16>
  %53 = sub nsw <16 x i16> %50, %52
  %54 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %53, i1 false)
  %55 = zext <16 x i16> %54 to <16 x i32>
  %56 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %55)
  %57 = add i32 %56, %46
  %58 = getelementptr inbounds nuw i8, ptr %0, i64 80
  %59 = getelementptr inbounds i8, ptr %48, i64 %5
  %60 = load <16 x i8>, ptr %58, align 1, !tbaa !6
  %61 = sext <16 x i8> %60 to <16 x i16>
  %62 = load <16 x i8>, ptr %59, align 1, !tbaa !6
  %63 = sext <16 x i8> %62 to <16 x i16>
  %64 = sub nsw <16 x i16> %61, %63
  %65 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %64, i1 false)
  %66 = zext <16 x i16> %65 to <16 x i32>
  %67 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %66)
  %68 = add i32 %67, %57
  %69 = getelementptr inbounds nuw i8, ptr %0, i64 96
  %70 = getelementptr inbounds i8, ptr %59, i64 %5
  %71 = load <16 x i8>, ptr %69, align 1, !tbaa !6
  %72 = sext <16 x i8> %71 to <16 x i16>
  %73 = load <16 x i8>, ptr %70, align 1, !tbaa !6
  %74 = sext <16 x i8> %73 to <16 x i16>
  %75 = sub nsw <16 x i16> %72, %74
  %76 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %75, i1 false)
  %77 = zext <16 x i16> %76 to <16 x i32>
  %78 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %77)
  %79 = add i32 %78, %68
  %80 = getelementptr inbounds nuw i8, ptr %0, i64 112
  %81 = getelementptr inbounds i8, ptr %70, i64 %5
  %82 = load <16 x i8>, ptr %80, align 1, !tbaa !6
  %83 = sext <16 x i8> %82 to <16 x i16>
  %84 = load <16 x i8>, ptr %81, align 1, !tbaa !6
  %85 = sext <16 x i8> %84 to <16 x i16>
  %86 = sub nsw <16 x i16> %83, %85
  %87 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %86, i1 false)
  %88 = zext <16 x i16> %87 to <16 x i32>
  %89 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %88)
  %90 = add i32 %89, %79
  %91 = getelementptr inbounds nuw i8, ptr %0, i64 128
  %92 = getelementptr inbounds i8, ptr %81, i64 %5
  %93 = load <16 x i8>, ptr %91, align 1, !tbaa !6
  %94 = sext <16 x i8> %93 to <16 x i16>
  %95 = load <16 x i8>, ptr %92, align 1, !tbaa !6
  %96 = sext <16 x i8> %95 to <16 x i16>
  %97 = sub nsw <16 x i16> %94, %96
  %98 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %97, i1 false)
  %99 = zext <16 x i16> %98 to <16 x i32>
  %100 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %99)
  %101 = add i32 %100, %90
  %102 = getelementptr inbounds nuw i8, ptr %0, i64 144
  %103 = getelementptr inbounds i8, ptr %92, i64 %5
  %104 = load <16 x i8>, ptr %102, align 1, !tbaa !6
  %105 = sext <16 x i8> %104 to <16 x i16>
  %106 = load <16 x i8>, ptr %103, align 1, !tbaa !6
  %107 = sext <16 x i8> %106 to <16 x i16>
  %108 = sub nsw <16 x i16> %105, %107
  %109 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %108, i1 false)
  %110 = zext <16 x i16> %109 to <16 x i32>
  %111 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %110)
  %112 = add i32 %111, %101
  %113 = getelementptr inbounds nuw i8, ptr %0, i64 160
  %114 = getelementptr inbounds i8, ptr %103, i64 %5
  %115 = load <16 x i8>, ptr %113, align 1, !tbaa !6
  %116 = sext <16 x i8> %115 to <16 x i16>
  %117 = load <16 x i8>, ptr %114, align 1, !tbaa !6
  %118 = sext <16 x i8> %117 to <16 x i16>
  %119 = sub nsw <16 x i16> %116, %118
  %120 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %119, i1 false)
  %121 = zext <16 x i16> %120 to <16 x i32>
  %122 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %121)
  %123 = add i32 %122, %112
  %124 = getelementptr inbounds nuw i8, ptr %0, i64 176
  %125 = getelementptr inbounds i8, ptr %114, i64 %5
  %126 = load <16 x i8>, ptr %124, align 1, !tbaa !6
  %127 = sext <16 x i8> %126 to <16 x i16>
  %128 = load <16 x i8>, ptr %125, align 1, !tbaa !6
  %129 = sext <16 x i8> %128 to <16 x i16>
  %130 = sub nsw <16 x i16> %127, %129
  %131 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %130, i1 false)
  %132 = zext <16 x i16> %131 to <16 x i32>
  %133 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %132)
  %134 = add i32 %133, %123
  %135 = getelementptr inbounds nuw i8, ptr %0, i64 192
  %136 = getelementptr inbounds i8, ptr %125, i64 %5
  %137 = load <16 x i8>, ptr %135, align 1, !tbaa !6
  %138 = sext <16 x i8> %137 to <16 x i16>
  %139 = load <16 x i8>, ptr %136, align 1, !tbaa !6
  %140 = sext <16 x i8> %139 to <16 x i16>
  %141 = sub nsw <16 x i16> %138, %140
  %142 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %141, i1 false)
  %143 = zext <16 x i16> %142 to <16 x i32>
  %144 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %143)
  %145 = add i32 %144, %134
  %146 = getelementptr inbounds nuw i8, ptr %0, i64 208
  %147 = getelementptr inbounds i8, ptr %136, i64 %5
  %148 = load <16 x i8>, ptr %146, align 1, !tbaa !6
  %149 = sext <16 x i8> %148 to <16 x i16>
  %150 = load <16 x i8>, ptr %147, align 1, !tbaa !6
  %151 = sext <16 x i8> %150 to <16 x i16>
  %152 = sub nsw <16 x i16> %149, %151
  %153 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %152, i1 false)
  %154 = zext <16 x i16> %153 to <16 x i32>
  %155 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %154)
  %156 = add i32 %155, %145
  %157 = getelementptr inbounds nuw i8, ptr %0, i64 224
  %158 = getelementptr inbounds i8, ptr %147, i64 %5
  %159 = load <16 x i8>, ptr %157, align 1, !tbaa !6
  %160 = sext <16 x i8> %159 to <16 x i16>
  %161 = load <16 x i8>, ptr %158, align 1, !tbaa !6
  %162 = sext <16 x i8> %161 to <16 x i16>
  %163 = sub nsw <16 x i16> %160, %162
  %164 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %163, i1 false)
  %165 = zext <16 x i16> %164 to <16 x i32>
  %166 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %165)
  %167 = add i32 %166, %156
  %168 = getelementptr inbounds nuw i8, ptr %0, i64 240
  %169 = getelementptr inbounds i8, ptr %158, i64 %5
  %170 = load <16 x i8>, ptr %168, align 1, !tbaa !6
  %171 = sext <16 x i8> %170 to <16 x i16>
  %172 = load <16 x i8>, ptr %169, align 1, !tbaa !6
  %173 = sext <16 x i8> %172 to <16 x i16>
  %174 = sub nsw <16 x i16> %171, %173
  %175 = tail call <16 x i16> @llvm.abs.v16i16(<16 x i16> %174, i1 false)
  %176 = zext <16 x i16> %175 to <16 x i32>
  %177 = tail call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %176)
  %178 = add i32 %177, %167
  store i32 %178, ptr %3, align 4, !tbaa !9
  ret void
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #1 {
  %1 = alloca [256 x i8], align 1
  %2 = alloca [256 x i8], align 1
  %3 = alloca i32, align 4
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.start.p0(ptr nonnull %3) #5
  br label %4

4:                                                ; preds = %4, %0
  %5 = phi i64 [ 0, %0 ], [ %24, %4 ]
  %6 = phi <16 x i32> [ <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>, %0 ], [ %25, %4 ]
  %7 = phi <16 x i8> [ <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, %0 ], [ %26, %4 ]
  %8 = and <16 x i32> %6, splat (i32 1)
  %9 = icmp eq <16 x i32> %8, zeroinitializer
  %10 = and <16 x i32> %6, splat (i32 7)
  %11 = trunc nuw nsw <16 x i32> %10 to <16 x i8>
  %12 = shl nuw nsw <16 x i8> %11, splat (i8 1)
  %13 = sub nuw nsw <16 x i8> splat (i8 -2), %12
  %14 = lshr <16 x i32> %10, splat (i32 1)
  %15 = trunc nuw nsw <16 x i32> %14 to <16 x i8>
  %16 = and <16 x i8> %7, splat (i8 6)
  %17 = shl nuw nsw <16 x i8> %16, splat (i8 1)
  %18 = or disjoint <16 x i8> %17, splat (i8 1)
  %19 = select <16 x i1> %9, <16 x i8> %18, <16 x i8> %13
  %20 = select <16 x i1> %9, <16 x i8> %16, <16 x i8> %15
  %21 = sub nsw <16 x i8> zeroinitializer, %20
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 %5
  store <16 x i8> %19, ptr %22, align 1, !tbaa !6
  %23 = getelementptr inbounds nuw i8, ptr %2, i64 %5
  store <16 x i8> %21, ptr %23, align 1, !tbaa !6
  %24 = add nuw i64 %5, 16
  %25 = add <16 x i32> %6, splat (i32 16)
  %26 = add <16 x i8> %7, splat (i8 16)
  %27 = icmp eq i64 %24, 256
  br i1 %27, label %28, label %4, !llvm.loop !11

28:                                               ; preds = %4
  call void @bar(ptr noundef nonnull %1, ptr noundef nonnull %2, i32 noundef 16, ptr noundef nonnull %3)
  %29 = load i32, ptr %3, align 4, !tbaa !9
  %30 = icmp eq i32 %29, 2368
  br i1 %30, label %32, label %31

31:                                               ; preds = %28
  tail call void @abort() #6
  unreachable

32:                                               ; preds = %28
  call void @llvm.lifetime.end.p0(ptr nonnull %3) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %2) #5
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #5
  ret i32 0
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #2

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare <16 x i16> @llvm.abs.v16i16(<16 x i16>, i1 immarg) #4

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.vector.reduce.add.v16i32(<16 x i32>) #4

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C/C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
!11 = distinct !{!11, !12, !13, !14}
!12 = !{!"llvm.loop.mustprogress"}
!13 = !{!"llvm.loop.isvectorized", i32 1}
!14 = !{!"llvm.loop.unroll.runtime.disable"}
