; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20060420-1.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/20060420-1.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@buffer = dso_local global [64 x float] zeroinitializer, align 4

; Function Attrs: nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable
define dso_local void @foo(ptr noundef %0, ptr noundef readonly captures(none) %1, i32 noundef %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = ptrtoint ptr %0 to i64
  %6 = icmp sgt i32 %3, 0
  br i1 %6, label %7, label %52

7:                                                ; preds = %4
  %8 = ptrtoint ptr %0 to i64
  %9 = icmp sgt i32 %2, 1
  %10 = sub i64 0, %8
  %11 = and i64 %10, 15
  %12 = zext nneg i32 %3 to i64
  br i1 %9, label %13, label %36

13:                                               ; preds = %7
  %14 = zext nneg i32 %2 to i64
  br label %15

15:                                               ; preds = %13, %32
  %16 = phi i64 [ 0, %13 ], [ %34, %32 ]
  %17 = icmp eq i64 %16, %11
  br i1 %17, label %46, label %18

18:                                               ; preds = %15
  %19 = load ptr, ptr %1, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw float, ptr %19, i64 %16
  %21 = load float, ptr %20, align 4, !tbaa !11
  br label %22

22:                                               ; preds = %18, %22
  %23 = phi i64 [ 1, %18 ], [ %30, %22 ]
  %24 = phi float [ %21, %18 ], [ %29, %22 ]
  %25 = getelementptr inbounds nuw ptr, ptr %1, i64 %23
  %26 = load ptr, ptr %25, align 8, !tbaa !6
  %27 = getelementptr inbounds nuw float, ptr %26, i64 %16
  %28 = load float, ptr %27, align 4, !tbaa !11
  %29 = fadd float %24, %28
  %30 = add nuw nsw i64 %23, 1
  %31 = icmp eq i64 %30, %14
  br i1 %31, label %32, label %22, !llvm.loop !13

32:                                               ; preds = %22
  %33 = getelementptr inbounds nuw float, ptr %0, i64 %16
  store float %29, ptr %33, align 4, !tbaa !11
  %34 = add nuw nsw i64 %16, 1
  %35 = icmp eq i64 %34, %12
  br i1 %35, label %46, label %15, !llvm.loop !15

36:                                               ; preds = %7, %39
  %37 = phi i64 [ %44, %39 ], [ 0, %7 ]
  %38 = icmp eq i64 %37, %11
  br i1 %38, label %49, label %39

39:                                               ; preds = %36
  %40 = load ptr, ptr %1, align 8, !tbaa !6
  %41 = getelementptr inbounds nuw float, ptr %40, i64 %37
  %42 = load float, ptr %41, align 4, !tbaa !11
  %43 = getelementptr inbounds nuw float, ptr %0, i64 %37
  store float %42, ptr %43, align 4, !tbaa !11
  %44 = add nuw nsw i64 %37, 1
  %45 = icmp eq i64 %44, %12
  br i1 %45, label %49, label %36, !llvm.loop !15

46:                                               ; preds = %32, %15
  %47 = phi i64 [ %16, %15 ], [ %34, %32 ]
  %48 = trunc i64 %47 to i32
  br label %52

49:                                               ; preds = %39, %36
  %50 = phi i64 [ %37, %36 ], [ %44, %39 ]
  %51 = trunc i64 %50 to i32
  br label %52

52:                                               ; preds = %49, %46, %4
  %53 = phi i32 [ 0, %4 ], [ %48, %46 ], [ %51, %49 ]
  %54 = add nsw i32 %3, -15
  %55 = icmp slt i32 %53, %54
  br i1 %55, label %56, label %106

56:                                               ; preds = %52
  %57 = icmp sgt i32 %2, 1
  %58 = zext nneg i32 %53 to i64
  %59 = sext i32 %54 to i64
  br i1 %57, label %60, label %159

60:                                               ; preds = %56
  %61 = zext nneg i32 %2 to i64
  br label %62

62:                                               ; preds = %60, %95
  %63 = phi i64 [ %58, %60 ], [ %100, %95 ]
  %64 = load ptr, ptr %1, align 8, !tbaa !6
  %65 = getelementptr inbounds nuw float, ptr %64, i64 %63
  %66 = load <4 x float>, ptr %65, align 16, !tbaa !16
  %67 = getelementptr inbounds nuw i8, ptr %65, i64 16
  %68 = load <4 x float>, ptr %67, align 16, !tbaa !16
  %69 = getelementptr inbounds nuw i8, ptr %65, i64 32
  %70 = load <4 x float>, ptr %69, align 16, !tbaa !16
  %71 = getelementptr inbounds nuw i8, ptr %65, i64 48
  %72 = load <4 x float>, ptr %71, align 16, !tbaa !16
  br label %73

73:                                               ; preds = %62, %73
  %74 = phi i64 [ 1, %62 ], [ %93, %73 ]
  %75 = phi <4 x float> [ %72, %62 ], [ %92, %73 ]
  %76 = phi <4 x float> [ %70, %62 ], [ %89, %73 ]
  %77 = phi <4 x float> [ %68, %62 ], [ %86, %73 ]
  %78 = phi <4 x float> [ %66, %62 ], [ %83, %73 ]
  %79 = getelementptr inbounds nuw ptr, ptr %1, i64 %74
  %80 = load ptr, ptr %79, align 8, !tbaa !6
  %81 = getelementptr inbounds nuw float, ptr %80, i64 %63
  %82 = load <4 x float>, ptr %81, align 16, !tbaa !16
  %83 = fadd <4 x float> %78, %82
  %84 = getelementptr inbounds nuw i8, ptr %81, i64 16
  %85 = load <4 x float>, ptr %84, align 16, !tbaa !16
  %86 = fadd <4 x float> %77, %85
  %87 = getelementptr inbounds nuw i8, ptr %81, i64 32
  %88 = load <4 x float>, ptr %87, align 16, !tbaa !16
  %89 = fadd <4 x float> %76, %88
  %90 = getelementptr inbounds nuw i8, ptr %81, i64 48
  %91 = load <4 x float>, ptr %90, align 16, !tbaa !16
  %92 = fadd <4 x float> %75, %91
  %93 = add nuw nsw i64 %74, 1
  %94 = icmp eq i64 %93, %61
  br i1 %94, label %95, label %73, !llvm.loop !17

95:                                               ; preds = %73
  %96 = getelementptr inbounds nuw float, ptr %0, i64 %63
  store <4 x float> %83, ptr %96, align 16, !tbaa !16
  %97 = getelementptr inbounds nuw i8, ptr %96, i64 16
  store <4 x float> %86, ptr %97, align 16, !tbaa !16
  %98 = getelementptr inbounds nuw i8, ptr %96, i64 32
  store <4 x float> %89, ptr %98, align 16, !tbaa !16
  %99 = getelementptr inbounds nuw i8, ptr %96, i64 48
  store <4 x float> %92, ptr %99, align 16, !tbaa !16
  %100 = add nuw nsw i64 %63, 16
  %101 = icmp slt i64 %100, %59
  br i1 %101, label %62, label %102, !llvm.loop !18

102:                                              ; preds = %95
  %103 = trunc nuw nsw i64 %100 to i32
  br label %106

104:                                              ; preds = %159
  %105 = trunc nuw nsw i64 %174 to i32
  br label %106

106:                                              ; preds = %104, %102, %52
  %107 = phi i32 [ %53, %52 ], [ %103, %102 ], [ %105, %104 ]
  %108 = icmp slt i32 %107, %3
  br i1 %108, label %109, label %183

109:                                              ; preds = %106
  %110 = load ptr, ptr %1, align 8, !tbaa !6
  %111 = icmp sgt i32 %2, 1
  %112 = zext i32 %107 to i64
  %113 = zext i32 %3 to i64
  br i1 %111, label %139, label %114

114:                                              ; preds = %109
  %115 = ptrtoint ptr %110 to i64
  %116 = sub nsw i64 %113, %112
  %117 = icmp ult i64 %116, 8
  %118 = sub i64 %5, %115
  %119 = icmp ult i64 %118, 32
  %120 = select i1 %117, i1 true, i1 %119
  br i1 %120, label %137, label %121

121:                                              ; preds = %114
  %122 = and i64 %116, -8
  %123 = add nsw i64 %122, %112
  br label %124

124:                                              ; preds = %124, %121
  %125 = phi i64 [ 0, %121 ], [ %133, %124 ]
  %126 = add i64 %125, %112
  %127 = getelementptr inbounds nuw float, ptr %110, i64 %126
  %128 = getelementptr inbounds nuw i8, ptr %127, i64 16
  %129 = load <4 x float>, ptr %127, align 4, !tbaa !11
  %130 = load <4 x float>, ptr %128, align 4, !tbaa !11
  %131 = getelementptr inbounds nuw float, ptr %0, i64 %126
  %132 = getelementptr inbounds nuw i8, ptr %131, i64 16
  store <4 x float> %129, ptr %131, align 4, !tbaa !11
  store <4 x float> %130, ptr %132, align 4, !tbaa !11
  %133 = add nuw i64 %125, 8
  %134 = icmp eq i64 %133, %122
  br i1 %134, label %135, label %124, !llvm.loop !19

135:                                              ; preds = %124
  %136 = icmp eq i64 %116, %122
  br i1 %136, label %183, label %137

137:                                              ; preds = %114, %135
  %138 = phi i64 [ %112, %114 ], [ %123, %135 ]
  br label %176

139:                                              ; preds = %109
  %140 = zext nneg i32 %2 to i64
  br label %141

141:                                              ; preds = %139, %155
  %142 = phi i64 [ %112, %139 ], [ %157, %155 ]
  %143 = getelementptr inbounds nuw float, ptr %110, i64 %142
  %144 = load float, ptr %143, align 4, !tbaa !11
  br label %145

145:                                              ; preds = %141, %145
  %146 = phi i64 [ 1, %141 ], [ %153, %145 ]
  %147 = phi float [ %144, %141 ], [ %152, %145 ]
  %148 = getelementptr inbounds nuw ptr, ptr %1, i64 %146
  %149 = load ptr, ptr %148, align 8, !tbaa !6
  %150 = getelementptr inbounds nuw float, ptr %149, i64 %142
  %151 = load float, ptr %150, align 4, !tbaa !11
  %152 = fadd float %147, %151
  %153 = add nuw nsw i64 %146, 1
  %154 = icmp eq i64 %153, %140
  br i1 %154, label %155, label %145, !llvm.loop !22

155:                                              ; preds = %145
  %156 = getelementptr inbounds nuw float, ptr %0, i64 %142
  store float %152, ptr %156, align 4, !tbaa !11
  %157 = add nuw nsw i64 %142, 1
  %158 = icmp eq i64 %157, %113
  br i1 %158, label %183, label %141, !llvm.loop !23

159:                                              ; preds = %56, %159
  %160 = phi i64 [ %174, %159 ], [ %58, %56 ]
  %161 = load ptr, ptr %1, align 8, !tbaa !6
  %162 = getelementptr inbounds nuw float, ptr %161, i64 %160
  %163 = load <4 x float>, ptr %162, align 16, !tbaa !16
  %164 = getelementptr inbounds nuw i8, ptr %162, i64 16
  %165 = load <4 x float>, ptr %164, align 16, !tbaa !16
  %166 = getelementptr inbounds nuw i8, ptr %162, i64 32
  %167 = load <4 x float>, ptr %166, align 16, !tbaa !16
  %168 = getelementptr inbounds nuw i8, ptr %162, i64 48
  %169 = load <4 x float>, ptr %168, align 16, !tbaa !16
  %170 = getelementptr inbounds nuw float, ptr %0, i64 %160
  store <4 x float> %163, ptr %170, align 16, !tbaa !16
  %171 = getelementptr inbounds nuw i8, ptr %170, i64 16
  store <4 x float> %165, ptr %171, align 16, !tbaa !16
  %172 = getelementptr inbounds nuw i8, ptr %170, i64 32
  store <4 x float> %167, ptr %172, align 16, !tbaa !16
  %173 = getelementptr inbounds nuw i8, ptr %170, i64 48
  store <4 x float> %169, ptr %173, align 16, !tbaa !16
  %174 = add nuw nsw i64 %160, 16
  %175 = icmp slt i64 %174, %59
  br i1 %175, label %159, label %104, !llvm.loop !18

176:                                              ; preds = %137, %176
  %177 = phi i64 [ %181, %176 ], [ %138, %137 ]
  %178 = getelementptr inbounds nuw float, ptr %110, i64 %177
  %179 = load float, ptr %178, align 4, !tbaa !11
  %180 = getelementptr inbounds nuw float, ptr %0, i64 %177
  store float %179, ptr %180, align 4, !tbaa !11
  %181 = add nuw nsw i64 %177, 1
  %182 = icmp eq i64 %181, %113
  br i1 %182, label %183, label %176, !llvm.loop !24

183:                                              ; preds = %176, %155, %135, %106
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #1

; Function Attrs: mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #1

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  %1 = alloca [2 x ptr], align 8
  call void @llvm.lifetime.start.p0(ptr nonnull %1) #4
  %2 = and i64 sub (i64 0, i64 ptrtoint (ptr @buffer to i64)), 60
  %3 = getelementptr inbounds nuw i8, ptr @buffer, i64 %2
  %4 = getelementptr inbounds nuw i8, ptr %3, i64 64
  store ptr %4, ptr %1, align 8, !tbaa !6
  %5 = getelementptr inbounds nuw i8, ptr %3, i64 128
  %6 = getelementptr inbounds nuw i8, ptr %1, i64 8
  store ptr %5, ptr %6, align 8, !tbaa !6
  store <4 x float> <float 0.000000e+00, float 1.200000e+01, float 2.400000e+01, float 3.600000e+01>, ptr %4, align 4, !tbaa !11
  store <4 x float> <float 0.000000e+00, float 1.300000e+01, float 2.600000e+01, float 3.900000e+01>, ptr %5, align 4, !tbaa !11
  %7 = getelementptr inbounds nuw i8, ptr %3, i64 80
  %8 = getelementptr inbounds nuw i8, ptr %3, i64 144
  store <4 x float> <float 4.800000e+01, float 6.000000e+01, float 7.200000e+01, float 8.400000e+01>, ptr %7, align 4, !tbaa !11
  store <4 x float> <float 5.200000e+01, float 6.500000e+01, float 7.800000e+01, float 9.100000e+01>, ptr %8, align 4, !tbaa !11
  %9 = getelementptr inbounds nuw i8, ptr %3, i64 96
  %10 = getelementptr inbounds nuw i8, ptr %3, i64 160
  store <4 x float> <float 9.600000e+01, float 1.080000e+02, float 1.200000e+02, float 1.320000e+02>, ptr %9, align 4, !tbaa !11
  store <4 x float> <float 1.040000e+02, float 1.170000e+02, float 1.300000e+02, float 1.430000e+02>, ptr %10, align 4, !tbaa !11
  %11 = getelementptr inbounds nuw i8, ptr %3, i64 112
  %12 = getelementptr inbounds nuw i8, ptr %3, i64 176
  store <4 x float> <float 1.440000e+02, float 1.560000e+02, float 1.680000e+02, float 1.800000e+02>, ptr %11, align 4, !tbaa !11
  store <4 x float> <float 1.560000e+02, float 1.690000e+02, float 1.820000e+02, float 1.950000e+02>, ptr %12, align 4, !tbaa !11
  call void @foo(ptr noundef nonnull %3, ptr noundef nonnull %1, i32 noundef 2, i32 noundef 16)
  %13 = load float, ptr %3, align 4, !tbaa !11
  %14 = fcmp une float %13, 0.000000e+00
  br i1 %14, label %76, label %15

15:                                               ; preds = %0
  %16 = getelementptr inbounds nuw i8, ptr %3, i64 4
  %17 = load float, ptr %16, align 4, !tbaa !11
  %18 = fcmp une float %17, 2.500000e+01
  br i1 %18, label %76, label %19

19:                                               ; preds = %15
  %20 = getelementptr inbounds nuw i8, ptr %3, i64 8
  %21 = load float, ptr %20, align 4, !tbaa !11
  %22 = fcmp une float %21, 5.000000e+01
  br i1 %22, label %76, label %23

23:                                               ; preds = %19
  %24 = getelementptr inbounds nuw i8, ptr %3, i64 12
  %25 = load float, ptr %24, align 4, !tbaa !11
  %26 = fcmp une float %25, 7.500000e+01
  br i1 %26, label %76, label %27

27:                                               ; preds = %23
  %28 = getelementptr inbounds nuw i8, ptr %3, i64 16
  %29 = load float, ptr %28, align 4, !tbaa !11
  %30 = fcmp une float %29, 1.000000e+02
  br i1 %30, label %76, label %31

31:                                               ; preds = %27
  %32 = getelementptr inbounds nuw i8, ptr %3, i64 20
  %33 = load float, ptr %32, align 4, !tbaa !11
  %34 = fcmp une float %33, 1.250000e+02
  br i1 %34, label %76, label %35

35:                                               ; preds = %31
  %36 = getelementptr inbounds nuw i8, ptr %3, i64 24
  %37 = load float, ptr %36, align 4, !tbaa !11
  %38 = fcmp une float %37, 1.500000e+02
  br i1 %38, label %76, label %39

39:                                               ; preds = %35
  %40 = getelementptr inbounds nuw i8, ptr %3, i64 28
  %41 = load float, ptr %40, align 4, !tbaa !11
  %42 = fcmp une float %41, 1.750000e+02
  br i1 %42, label %76, label %43

43:                                               ; preds = %39
  %44 = getelementptr inbounds nuw i8, ptr %3, i64 32
  %45 = load float, ptr %44, align 4, !tbaa !11
  %46 = fcmp une float %45, 2.000000e+02
  br i1 %46, label %76, label %47

47:                                               ; preds = %43
  %48 = getelementptr inbounds nuw i8, ptr %3, i64 36
  %49 = load float, ptr %48, align 4, !tbaa !11
  %50 = fcmp une float %49, 2.250000e+02
  br i1 %50, label %76, label %51

51:                                               ; preds = %47
  %52 = getelementptr inbounds nuw i8, ptr %3, i64 40
  %53 = load float, ptr %52, align 4, !tbaa !11
  %54 = fcmp une float %53, 2.500000e+02
  br i1 %54, label %76, label %55

55:                                               ; preds = %51
  %56 = getelementptr inbounds nuw i8, ptr %3, i64 44
  %57 = load float, ptr %56, align 4, !tbaa !11
  %58 = fcmp une float %57, 2.750000e+02
  br i1 %58, label %76, label %59

59:                                               ; preds = %55
  %60 = getelementptr inbounds nuw i8, ptr %3, i64 48
  %61 = load float, ptr %60, align 4, !tbaa !11
  %62 = fcmp une float %61, 3.000000e+02
  br i1 %62, label %76, label %63

63:                                               ; preds = %59
  %64 = getelementptr inbounds nuw i8, ptr %3, i64 52
  %65 = load float, ptr %64, align 4, !tbaa !11
  %66 = fcmp une float %65, 3.250000e+02
  br i1 %66, label %76, label %67

67:                                               ; preds = %63
  %68 = getelementptr inbounds nuw i8, ptr %3, i64 56
  %69 = load float, ptr %68, align 4, !tbaa !11
  %70 = fcmp une float %69, 3.500000e+02
  br i1 %70, label %76, label %71

71:                                               ; preds = %67
  %72 = getelementptr inbounds nuw i8, ptr %3, i64 60
  %73 = load float, ptr %72, align 4, !tbaa !11
  %74 = fcmp une float %73, 3.750000e+02
  br i1 %74, label %76, label %75

75:                                               ; preds = %71
  call void @llvm.lifetime.end.p0(ptr nonnull %1) #4
  ret i32 0

76:                                               ; preds = %71, %67, %63, %59, %55, %51, %47, %43, %39, %35, %31, %27, %23, %19, %15, %0
  tail call void @abort() #5
  unreachable
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

attributes #0 = { nofree noinline norecurse nosync nounwind memory(read, argmem: readwrite, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nounwind }
attributes #5 = { noreturn nounwind }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"clang version 22.0.0git (https://github.com/steven-studio/llvm-project.git c2901ea177a93cdcea513ae5bdc6a189f274f4ca)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"p1 float", !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{!12, !12, i64 0}
!12 = !{!"float", !9, i64 0}
!13 = distinct !{!13, !14}
!14 = !{!"llvm.loop.mustprogress"}
!15 = distinct !{!15, !14}
!16 = !{!9, !9, i64 0}
!17 = distinct !{!17, !14}
!18 = distinct !{!18, !14}
!19 = distinct !{!19, !14, !20, !21}
!20 = !{!"llvm.loop.isvectorized", i32 1}
!21 = !{!"llvm.loop.unroll.runtime.disable"}
!22 = distinct !{!22, !14}
!23 = distinct !{!23, !14}
!24 = distinct !{!24, !14, !20}
