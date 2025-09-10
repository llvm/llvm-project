; ModuleID = '/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr91137.c'
source_filename = "/home/youzhewei.linux/work/llvm-project/llvm-test-suite/SingleSource/Regression/C/gcc-c-torture/execute/pr91137.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-gnu"

@b = dso_local local_unnamed_addr global i32 0, align 4
@c = dso_local local_unnamed_addr global [70 x i32] zeroinitializer, align 16
@d = dso_local local_unnamed_addr global [70 x [70 x i32]] zeroinitializer, align 4
@e = dso_local local_unnamed_addr global i32 0, align 4
@a = dso_local global i64 0, align 8

; Function Attrs: mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable
define dso_local void @f(ptr noundef writeonly captures(none) initializes((0, 8)) %0, i32 noundef %1) local_unnamed_addr #0 {
  %3 = sext i32 %1 to i64
  store i64 %3, ptr %0, align 8, !tbaa !6
  ret void
}

; Function Attrs: nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable
define dso_local void @fn2() local_unnamed_addr #1 {
  %1 = load i32, ptr @b, align 4, !tbaa !10
  %2 = icmp eq i32 %1, 0
  br i1 %2, label %3, label %155

3:                                                ; preds = %0
  %4 = load i32, ptr @c, align 4, !tbaa !10
  br label %5

5:                                                ; preds = %152, %3
  %6 = phi i32 [ 0, %3 ], [ %153, %152 ]
  br label %7

7:                                                ; preds = %7, %5
  %8 = phi i64 [ %150, %7 ], [ 0, %5 ]
  %9 = getelementptr inbounds nuw i32, ptr @d, i64 %8
  %10 = getelementptr inbounds nuw [70 x i32], ptr @d, i64 %8, i64 1
  %11 = load i32, ptr %9, align 4, !tbaa !10
  store i32 %11, ptr %10, align 4, !tbaa !10
  %12 = getelementptr inbounds nuw i8, ptr %9, i64 280
  %13 = load i32, ptr %12, align 4, !tbaa !10
  store i32 %13, ptr %10, align 4, !tbaa !10
  %14 = getelementptr inbounds nuw i8, ptr %9, i64 560
  %15 = load i32, ptr %14, align 4, !tbaa !10
  store i32 %15, ptr %10, align 4, !tbaa !10
  %16 = getelementptr inbounds nuw i8, ptr %9, i64 840
  %17 = load i32, ptr %16, align 4, !tbaa !10
  store i32 %17, ptr %10, align 4, !tbaa !10
  %18 = getelementptr inbounds nuw i8, ptr %9, i64 1120
  %19 = load i32, ptr %18, align 4, !tbaa !10
  store i32 %19, ptr %10, align 4, !tbaa !10
  %20 = getelementptr inbounds nuw i8, ptr %9, i64 1400
  %21 = load i32, ptr %20, align 4, !tbaa !10
  store i32 %21, ptr %10, align 4, !tbaa !10
  %22 = getelementptr inbounds nuw i8, ptr %9, i64 1680
  %23 = load i32, ptr %22, align 4, !tbaa !10
  store i32 %23, ptr %10, align 4, !tbaa !10
  %24 = getelementptr inbounds nuw i8, ptr %9, i64 1960
  %25 = load i32, ptr %24, align 4, !tbaa !10
  store i32 %25, ptr %10, align 4, !tbaa !10
  %26 = getelementptr inbounds nuw i8, ptr %9, i64 2240
  %27 = load i32, ptr %26, align 4, !tbaa !10
  store i32 %27, ptr %10, align 4, !tbaa !10
  %28 = getelementptr inbounds nuw i8, ptr %9, i64 2520
  %29 = load i32, ptr %28, align 4, !tbaa !10
  store i32 %29, ptr %10, align 4, !tbaa !10
  %30 = getelementptr inbounds nuw i8, ptr %9, i64 2800
  %31 = load i32, ptr %30, align 4, !tbaa !10
  store i32 %31, ptr %10, align 4, !tbaa !10
  %32 = getelementptr inbounds nuw i8, ptr %9, i64 3080
  %33 = load i32, ptr %32, align 4, !tbaa !10
  store i32 %33, ptr %10, align 4, !tbaa !10
  %34 = getelementptr inbounds nuw i8, ptr %9, i64 3360
  %35 = load i32, ptr %34, align 4, !tbaa !10
  store i32 %35, ptr %10, align 4, !tbaa !10
  %36 = getelementptr inbounds nuw i8, ptr %9, i64 3640
  %37 = load i32, ptr %36, align 4, !tbaa !10
  store i32 %37, ptr %10, align 4, !tbaa !10
  %38 = getelementptr inbounds nuw i8, ptr %9, i64 3920
  %39 = load i32, ptr %38, align 4, !tbaa !10
  store i32 %39, ptr %10, align 4, !tbaa !10
  %40 = getelementptr inbounds nuw i8, ptr %9, i64 4200
  %41 = load i32, ptr %40, align 4, !tbaa !10
  store i32 %41, ptr %10, align 4, !tbaa !10
  %42 = getelementptr inbounds nuw i8, ptr %9, i64 4480
  %43 = load i32, ptr %42, align 4, !tbaa !10
  store i32 %43, ptr %10, align 4, !tbaa !10
  %44 = getelementptr inbounds nuw i8, ptr %9, i64 4760
  %45 = load i32, ptr %44, align 4, !tbaa !10
  store i32 %45, ptr %10, align 4, !tbaa !10
  %46 = getelementptr inbounds nuw i8, ptr %9, i64 5040
  %47 = load i32, ptr %46, align 4, !tbaa !10
  store i32 %47, ptr %10, align 4, !tbaa !10
  %48 = getelementptr inbounds nuw i8, ptr %9, i64 5320
  %49 = load i32, ptr %48, align 4, !tbaa !10
  store i32 %49, ptr %10, align 4, !tbaa !10
  %50 = getelementptr inbounds nuw i8, ptr %9, i64 5600
  %51 = load i32, ptr %50, align 4, !tbaa !10
  store i32 %51, ptr %10, align 4, !tbaa !10
  %52 = getelementptr inbounds nuw i8, ptr %9, i64 5880
  %53 = load i32, ptr %52, align 4, !tbaa !10
  store i32 %53, ptr %10, align 4, !tbaa !10
  %54 = getelementptr inbounds nuw i8, ptr %9, i64 6160
  %55 = load i32, ptr %54, align 4, !tbaa !10
  store i32 %55, ptr %10, align 4, !tbaa !10
  %56 = getelementptr inbounds nuw i8, ptr %9, i64 6440
  %57 = load i32, ptr %56, align 4, !tbaa !10
  store i32 %57, ptr %10, align 4, !tbaa !10
  %58 = getelementptr inbounds nuw i8, ptr %9, i64 6720
  %59 = load i32, ptr %58, align 4, !tbaa !10
  store i32 %59, ptr %10, align 4, !tbaa !10
  %60 = getelementptr inbounds nuw i8, ptr %9, i64 7000
  %61 = load i32, ptr %60, align 4, !tbaa !10
  store i32 %61, ptr %10, align 4, !tbaa !10
  %62 = getelementptr inbounds nuw i8, ptr %9, i64 7280
  %63 = load i32, ptr %62, align 4, !tbaa !10
  store i32 %63, ptr %10, align 4, !tbaa !10
  %64 = getelementptr inbounds nuw i8, ptr %9, i64 7560
  %65 = load i32, ptr %64, align 4, !tbaa !10
  store i32 %65, ptr %10, align 4, !tbaa !10
  %66 = getelementptr inbounds nuw i8, ptr %9, i64 7840
  %67 = load i32, ptr %66, align 4, !tbaa !10
  store i32 %67, ptr %10, align 4, !tbaa !10
  %68 = getelementptr inbounds nuw i8, ptr %9, i64 8120
  %69 = load i32, ptr %68, align 4, !tbaa !10
  store i32 %69, ptr %10, align 4, !tbaa !10
  %70 = getelementptr inbounds nuw i8, ptr %9, i64 8400
  %71 = load i32, ptr %70, align 4, !tbaa !10
  store i32 %71, ptr %10, align 4, !tbaa !10
  %72 = getelementptr inbounds nuw i8, ptr %9, i64 8680
  %73 = load i32, ptr %72, align 4, !tbaa !10
  store i32 %73, ptr %10, align 4, !tbaa !10
  %74 = getelementptr inbounds nuw i8, ptr %9, i64 8960
  %75 = load i32, ptr %74, align 4, !tbaa !10
  store i32 %75, ptr %10, align 4, !tbaa !10
  %76 = getelementptr inbounds nuw i8, ptr %9, i64 9240
  %77 = load i32, ptr %76, align 4, !tbaa !10
  store i32 %77, ptr %10, align 4, !tbaa !10
  %78 = getelementptr inbounds nuw i8, ptr %9, i64 9520
  %79 = load i32, ptr %78, align 4, !tbaa !10
  store i32 %79, ptr %10, align 4, !tbaa !10
  %80 = getelementptr inbounds nuw i8, ptr %9, i64 9800
  %81 = load i32, ptr %80, align 4, !tbaa !10
  store i32 %81, ptr %10, align 4, !tbaa !10
  %82 = getelementptr inbounds nuw i8, ptr %9, i64 10080
  %83 = load i32, ptr %82, align 4, !tbaa !10
  store i32 %83, ptr %10, align 4, !tbaa !10
  %84 = getelementptr inbounds nuw i8, ptr %9, i64 10360
  %85 = load i32, ptr %84, align 4, !tbaa !10
  store i32 %85, ptr %10, align 4, !tbaa !10
  %86 = getelementptr inbounds nuw i8, ptr %9, i64 10640
  %87 = load i32, ptr %86, align 4, !tbaa !10
  store i32 %87, ptr %10, align 4, !tbaa !10
  %88 = getelementptr inbounds nuw i8, ptr %9, i64 10920
  %89 = load i32, ptr %88, align 4, !tbaa !10
  store i32 %89, ptr %10, align 4, !tbaa !10
  %90 = getelementptr inbounds nuw i8, ptr %9, i64 11200
  %91 = load i32, ptr %90, align 4, !tbaa !10
  store i32 %91, ptr %10, align 4, !tbaa !10
  %92 = getelementptr inbounds nuw i8, ptr %9, i64 11480
  %93 = load i32, ptr %92, align 4, !tbaa !10
  store i32 %93, ptr %10, align 4, !tbaa !10
  %94 = getelementptr inbounds nuw i8, ptr %9, i64 11760
  %95 = load i32, ptr %94, align 4, !tbaa !10
  store i32 %95, ptr %10, align 4, !tbaa !10
  %96 = getelementptr inbounds nuw i8, ptr %9, i64 12040
  %97 = load i32, ptr %96, align 4, !tbaa !10
  store i32 %97, ptr %10, align 4, !tbaa !10
  %98 = getelementptr inbounds nuw i8, ptr %9, i64 12320
  %99 = load i32, ptr %98, align 4, !tbaa !10
  store i32 %99, ptr %10, align 4, !tbaa !10
  %100 = getelementptr inbounds nuw i8, ptr %9, i64 12600
  %101 = load i32, ptr %100, align 4, !tbaa !10
  store i32 %101, ptr %10, align 4, !tbaa !10
  %102 = getelementptr inbounds nuw i8, ptr %9, i64 12880
  %103 = load i32, ptr %102, align 4, !tbaa !10
  store i32 %103, ptr %10, align 4, !tbaa !10
  %104 = getelementptr inbounds nuw i8, ptr %9, i64 13160
  %105 = load i32, ptr %104, align 4, !tbaa !10
  store i32 %105, ptr %10, align 4, !tbaa !10
  %106 = getelementptr inbounds nuw i8, ptr %9, i64 13440
  %107 = load i32, ptr %106, align 4, !tbaa !10
  store i32 %107, ptr %10, align 4, !tbaa !10
  %108 = getelementptr inbounds nuw i8, ptr %9, i64 13720
  %109 = load i32, ptr %108, align 4, !tbaa !10
  store i32 %109, ptr %10, align 4, !tbaa !10
  %110 = getelementptr inbounds nuw i8, ptr %9, i64 14000
  %111 = load i32, ptr %110, align 4, !tbaa !10
  store i32 %111, ptr %10, align 4, !tbaa !10
  %112 = getelementptr inbounds nuw i8, ptr %9, i64 14280
  %113 = load i32, ptr %112, align 4, !tbaa !10
  store i32 %113, ptr %10, align 4, !tbaa !10
  %114 = getelementptr inbounds nuw i8, ptr %9, i64 14560
  %115 = load i32, ptr %114, align 4, !tbaa !10
  store i32 %115, ptr %10, align 4, !tbaa !10
  %116 = getelementptr inbounds nuw i8, ptr %9, i64 14840
  %117 = load i32, ptr %116, align 4, !tbaa !10
  store i32 %117, ptr %10, align 4, !tbaa !10
  %118 = getelementptr inbounds nuw i8, ptr %9, i64 15120
  %119 = load i32, ptr %118, align 4, !tbaa !10
  store i32 %119, ptr %10, align 4, !tbaa !10
  %120 = getelementptr inbounds nuw i8, ptr %9, i64 15400
  %121 = load i32, ptr %120, align 4, !tbaa !10
  store i32 %121, ptr %10, align 4, !tbaa !10
  %122 = getelementptr inbounds nuw i8, ptr %9, i64 15680
  %123 = load i32, ptr %122, align 4, !tbaa !10
  store i32 %123, ptr %10, align 4, !tbaa !10
  %124 = getelementptr inbounds nuw i8, ptr %9, i64 15960
  %125 = load i32, ptr %124, align 4, !tbaa !10
  store i32 %125, ptr %10, align 4, !tbaa !10
  %126 = getelementptr inbounds nuw i8, ptr %9, i64 16240
  %127 = load i32, ptr %126, align 4, !tbaa !10
  store i32 %127, ptr %10, align 4, !tbaa !10
  %128 = getelementptr inbounds nuw i8, ptr %9, i64 16520
  %129 = load i32, ptr %128, align 4, !tbaa !10
  store i32 %129, ptr %10, align 4, !tbaa !10
  %130 = getelementptr inbounds nuw i8, ptr %9, i64 16800
  %131 = load i32, ptr %130, align 4, !tbaa !10
  store i32 %131, ptr %10, align 4, !tbaa !10
  %132 = getelementptr inbounds nuw i8, ptr %9, i64 17080
  %133 = load i32, ptr %132, align 4, !tbaa !10
  store i32 %133, ptr %10, align 4, !tbaa !10
  %134 = getelementptr inbounds nuw i8, ptr %9, i64 17360
  %135 = load i32, ptr %134, align 4, !tbaa !10
  store i32 %135, ptr %10, align 4, !tbaa !10
  %136 = getelementptr inbounds nuw i8, ptr %9, i64 17640
  %137 = load i32, ptr %136, align 4, !tbaa !10
  store i32 %137, ptr %10, align 4, !tbaa !10
  %138 = getelementptr inbounds nuw i8, ptr %9, i64 17920
  %139 = load i32, ptr %138, align 4, !tbaa !10
  store i32 %139, ptr %10, align 4, !tbaa !10
  %140 = getelementptr inbounds nuw i8, ptr %9, i64 18200
  %141 = load i32, ptr %140, align 4, !tbaa !10
  store i32 %141, ptr %10, align 4, !tbaa !10
  %142 = getelementptr inbounds nuw i8, ptr %9, i64 18480
  %143 = load i32, ptr %142, align 4, !tbaa !10
  store i32 %143, ptr %10, align 4, !tbaa !10
  %144 = getelementptr inbounds nuw i8, ptr %9, i64 18760
  %145 = load i32, ptr %144, align 4, !tbaa !10
  store i32 %145, ptr %10, align 4, !tbaa !10
  %146 = getelementptr inbounds nuw i8, ptr %9, i64 19040
  %147 = load i32, ptr %146, align 4, !tbaa !10
  store i32 %147, ptr %10, align 4, !tbaa !10
  %148 = getelementptr inbounds nuw i8, ptr %9, i64 19320
  %149 = load i32, ptr %148, align 4, !tbaa !10
  store i32 %149, ptr %10, align 4, !tbaa !10
  %150 = add nuw nsw i64 %8, 1
  %151 = icmp eq i64 %150, 70
  br i1 %151, label %152, label %7, !llvm.loop !12

152:                                              ; preds = %7
  %153 = add nuw nsw i32 %6, 1
  %154 = icmp eq i32 %153, 70
  br i1 %154, label %159, label %5, !llvm.loop !14

155:                                              ; preds = %0, %161
  %156 = phi i32 [ %162, %161 ], [ 0, %0 ]
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 4 dereferenceable(280) @c, i8 0, i64 280, i1 false), !tbaa !10
  br label %164

157:                                              ; preds = %161
  %158 = load i32, ptr @c, align 4, !tbaa !10
  br label %159

159:                                              ; preds = %152, %157
  %160 = phi i32 [ %158, %157 ], [ %4, %152 ]
  store i32 %160, ptr @e, align 4, !tbaa !10
  ret void

161:                                              ; preds = %164
  %162 = add nuw nsw i32 %156, 1
  %163 = icmp eq i32 %162, 70
  br i1 %163, label %157, label %155, !llvm.loop !14

164:                                              ; preds = %155, %164
  %165 = phi i64 [ 0, %155 ], [ %307, %164 ]
  %166 = getelementptr inbounds nuw i32, ptr @d, i64 %165
  %167 = getelementptr inbounds nuw [70 x i32], ptr @d, i64 %165, i64 1
  %168 = load i32, ptr %166, align 4, !tbaa !10
  store i32 %168, ptr %167, align 4, !tbaa !10
  %169 = getelementptr inbounds nuw i8, ptr %166, i64 280
  %170 = load i32, ptr %169, align 4, !tbaa !10
  store i32 %170, ptr %167, align 4, !tbaa !10
  %171 = getelementptr inbounds nuw i8, ptr %166, i64 560
  %172 = load i32, ptr %171, align 4, !tbaa !10
  store i32 %172, ptr %167, align 4, !tbaa !10
  %173 = getelementptr inbounds nuw i8, ptr %166, i64 840
  %174 = load i32, ptr %173, align 4, !tbaa !10
  store i32 %174, ptr %167, align 4, !tbaa !10
  %175 = getelementptr inbounds nuw i8, ptr %166, i64 1120
  %176 = load i32, ptr %175, align 4, !tbaa !10
  store i32 %176, ptr %167, align 4, !tbaa !10
  %177 = getelementptr inbounds nuw i8, ptr %166, i64 1400
  %178 = load i32, ptr %177, align 4, !tbaa !10
  store i32 %178, ptr %167, align 4, !tbaa !10
  %179 = getelementptr inbounds nuw i8, ptr %166, i64 1680
  %180 = load i32, ptr %179, align 4, !tbaa !10
  store i32 %180, ptr %167, align 4, !tbaa !10
  %181 = getelementptr inbounds nuw i8, ptr %166, i64 1960
  %182 = load i32, ptr %181, align 4, !tbaa !10
  store i32 %182, ptr %167, align 4, !tbaa !10
  %183 = getelementptr inbounds nuw i8, ptr %166, i64 2240
  %184 = load i32, ptr %183, align 4, !tbaa !10
  store i32 %184, ptr %167, align 4, !tbaa !10
  %185 = getelementptr inbounds nuw i8, ptr %166, i64 2520
  %186 = load i32, ptr %185, align 4, !tbaa !10
  store i32 %186, ptr %167, align 4, !tbaa !10
  %187 = getelementptr inbounds nuw i8, ptr %166, i64 2800
  %188 = load i32, ptr %187, align 4, !tbaa !10
  store i32 %188, ptr %167, align 4, !tbaa !10
  %189 = getelementptr inbounds nuw i8, ptr %166, i64 3080
  %190 = load i32, ptr %189, align 4, !tbaa !10
  store i32 %190, ptr %167, align 4, !tbaa !10
  %191 = getelementptr inbounds nuw i8, ptr %166, i64 3360
  %192 = load i32, ptr %191, align 4, !tbaa !10
  store i32 %192, ptr %167, align 4, !tbaa !10
  %193 = getelementptr inbounds nuw i8, ptr %166, i64 3640
  %194 = load i32, ptr %193, align 4, !tbaa !10
  store i32 %194, ptr %167, align 4, !tbaa !10
  %195 = getelementptr inbounds nuw i8, ptr %166, i64 3920
  %196 = load i32, ptr %195, align 4, !tbaa !10
  store i32 %196, ptr %167, align 4, !tbaa !10
  %197 = getelementptr inbounds nuw i8, ptr %166, i64 4200
  %198 = load i32, ptr %197, align 4, !tbaa !10
  store i32 %198, ptr %167, align 4, !tbaa !10
  %199 = getelementptr inbounds nuw i8, ptr %166, i64 4480
  %200 = load i32, ptr %199, align 4, !tbaa !10
  store i32 %200, ptr %167, align 4, !tbaa !10
  %201 = getelementptr inbounds nuw i8, ptr %166, i64 4760
  %202 = load i32, ptr %201, align 4, !tbaa !10
  store i32 %202, ptr %167, align 4, !tbaa !10
  %203 = getelementptr inbounds nuw i8, ptr %166, i64 5040
  %204 = load i32, ptr %203, align 4, !tbaa !10
  store i32 %204, ptr %167, align 4, !tbaa !10
  %205 = getelementptr inbounds nuw i8, ptr %166, i64 5320
  %206 = load i32, ptr %205, align 4, !tbaa !10
  store i32 %206, ptr %167, align 4, !tbaa !10
  %207 = getelementptr inbounds nuw i8, ptr %166, i64 5600
  %208 = load i32, ptr %207, align 4, !tbaa !10
  store i32 %208, ptr %167, align 4, !tbaa !10
  %209 = getelementptr inbounds nuw i8, ptr %166, i64 5880
  %210 = load i32, ptr %209, align 4, !tbaa !10
  store i32 %210, ptr %167, align 4, !tbaa !10
  %211 = getelementptr inbounds nuw i8, ptr %166, i64 6160
  %212 = load i32, ptr %211, align 4, !tbaa !10
  store i32 %212, ptr %167, align 4, !tbaa !10
  %213 = getelementptr inbounds nuw i8, ptr %166, i64 6440
  %214 = load i32, ptr %213, align 4, !tbaa !10
  store i32 %214, ptr %167, align 4, !tbaa !10
  %215 = getelementptr inbounds nuw i8, ptr %166, i64 6720
  %216 = load i32, ptr %215, align 4, !tbaa !10
  store i32 %216, ptr %167, align 4, !tbaa !10
  %217 = getelementptr inbounds nuw i8, ptr %166, i64 7000
  %218 = load i32, ptr %217, align 4, !tbaa !10
  store i32 %218, ptr %167, align 4, !tbaa !10
  %219 = getelementptr inbounds nuw i8, ptr %166, i64 7280
  %220 = load i32, ptr %219, align 4, !tbaa !10
  store i32 %220, ptr %167, align 4, !tbaa !10
  %221 = getelementptr inbounds nuw i8, ptr %166, i64 7560
  %222 = load i32, ptr %221, align 4, !tbaa !10
  store i32 %222, ptr %167, align 4, !tbaa !10
  %223 = getelementptr inbounds nuw i8, ptr %166, i64 7840
  %224 = load i32, ptr %223, align 4, !tbaa !10
  store i32 %224, ptr %167, align 4, !tbaa !10
  %225 = getelementptr inbounds nuw i8, ptr %166, i64 8120
  %226 = load i32, ptr %225, align 4, !tbaa !10
  store i32 %226, ptr %167, align 4, !tbaa !10
  %227 = getelementptr inbounds nuw i8, ptr %166, i64 8400
  %228 = load i32, ptr %227, align 4, !tbaa !10
  store i32 %228, ptr %167, align 4, !tbaa !10
  %229 = getelementptr inbounds nuw i8, ptr %166, i64 8680
  %230 = load i32, ptr %229, align 4, !tbaa !10
  store i32 %230, ptr %167, align 4, !tbaa !10
  %231 = getelementptr inbounds nuw i8, ptr %166, i64 8960
  %232 = load i32, ptr %231, align 4, !tbaa !10
  store i32 %232, ptr %167, align 4, !tbaa !10
  %233 = getelementptr inbounds nuw i8, ptr %166, i64 9240
  %234 = load i32, ptr %233, align 4, !tbaa !10
  store i32 %234, ptr %167, align 4, !tbaa !10
  %235 = getelementptr inbounds nuw i8, ptr %166, i64 9520
  %236 = load i32, ptr %235, align 4, !tbaa !10
  store i32 %236, ptr %167, align 4, !tbaa !10
  %237 = getelementptr inbounds nuw i8, ptr %166, i64 9800
  %238 = load i32, ptr %237, align 4, !tbaa !10
  store i32 %238, ptr %167, align 4, !tbaa !10
  %239 = getelementptr inbounds nuw i8, ptr %166, i64 10080
  %240 = load i32, ptr %239, align 4, !tbaa !10
  store i32 %240, ptr %167, align 4, !tbaa !10
  %241 = getelementptr inbounds nuw i8, ptr %166, i64 10360
  %242 = load i32, ptr %241, align 4, !tbaa !10
  store i32 %242, ptr %167, align 4, !tbaa !10
  %243 = getelementptr inbounds nuw i8, ptr %166, i64 10640
  %244 = load i32, ptr %243, align 4, !tbaa !10
  store i32 %244, ptr %167, align 4, !tbaa !10
  %245 = getelementptr inbounds nuw i8, ptr %166, i64 10920
  %246 = load i32, ptr %245, align 4, !tbaa !10
  store i32 %246, ptr %167, align 4, !tbaa !10
  %247 = getelementptr inbounds nuw i8, ptr %166, i64 11200
  %248 = load i32, ptr %247, align 4, !tbaa !10
  store i32 %248, ptr %167, align 4, !tbaa !10
  %249 = getelementptr inbounds nuw i8, ptr %166, i64 11480
  %250 = load i32, ptr %249, align 4, !tbaa !10
  store i32 %250, ptr %167, align 4, !tbaa !10
  %251 = getelementptr inbounds nuw i8, ptr %166, i64 11760
  %252 = load i32, ptr %251, align 4, !tbaa !10
  store i32 %252, ptr %167, align 4, !tbaa !10
  %253 = getelementptr inbounds nuw i8, ptr %166, i64 12040
  %254 = load i32, ptr %253, align 4, !tbaa !10
  store i32 %254, ptr %167, align 4, !tbaa !10
  %255 = getelementptr inbounds nuw i8, ptr %166, i64 12320
  %256 = load i32, ptr %255, align 4, !tbaa !10
  store i32 %256, ptr %167, align 4, !tbaa !10
  %257 = getelementptr inbounds nuw i8, ptr %166, i64 12600
  %258 = load i32, ptr %257, align 4, !tbaa !10
  store i32 %258, ptr %167, align 4, !tbaa !10
  %259 = getelementptr inbounds nuw i8, ptr %166, i64 12880
  %260 = load i32, ptr %259, align 4, !tbaa !10
  store i32 %260, ptr %167, align 4, !tbaa !10
  %261 = getelementptr inbounds nuw i8, ptr %166, i64 13160
  %262 = load i32, ptr %261, align 4, !tbaa !10
  store i32 %262, ptr %167, align 4, !tbaa !10
  %263 = getelementptr inbounds nuw i8, ptr %166, i64 13440
  %264 = load i32, ptr %263, align 4, !tbaa !10
  store i32 %264, ptr %167, align 4, !tbaa !10
  %265 = getelementptr inbounds nuw i8, ptr %166, i64 13720
  %266 = load i32, ptr %265, align 4, !tbaa !10
  store i32 %266, ptr %167, align 4, !tbaa !10
  %267 = getelementptr inbounds nuw i8, ptr %166, i64 14000
  %268 = load i32, ptr %267, align 4, !tbaa !10
  store i32 %268, ptr %167, align 4, !tbaa !10
  %269 = getelementptr inbounds nuw i8, ptr %166, i64 14280
  %270 = load i32, ptr %269, align 4, !tbaa !10
  store i32 %270, ptr %167, align 4, !tbaa !10
  %271 = getelementptr inbounds nuw i8, ptr %166, i64 14560
  %272 = load i32, ptr %271, align 4, !tbaa !10
  store i32 %272, ptr %167, align 4, !tbaa !10
  %273 = getelementptr inbounds nuw i8, ptr %166, i64 14840
  %274 = load i32, ptr %273, align 4, !tbaa !10
  store i32 %274, ptr %167, align 4, !tbaa !10
  %275 = getelementptr inbounds nuw i8, ptr %166, i64 15120
  %276 = load i32, ptr %275, align 4, !tbaa !10
  store i32 %276, ptr %167, align 4, !tbaa !10
  %277 = getelementptr inbounds nuw i8, ptr %166, i64 15400
  %278 = load i32, ptr %277, align 4, !tbaa !10
  store i32 %278, ptr %167, align 4, !tbaa !10
  %279 = getelementptr inbounds nuw i8, ptr %166, i64 15680
  %280 = load i32, ptr %279, align 4, !tbaa !10
  store i32 %280, ptr %167, align 4, !tbaa !10
  %281 = getelementptr inbounds nuw i8, ptr %166, i64 15960
  %282 = load i32, ptr %281, align 4, !tbaa !10
  store i32 %282, ptr %167, align 4, !tbaa !10
  %283 = getelementptr inbounds nuw i8, ptr %166, i64 16240
  %284 = load i32, ptr %283, align 4, !tbaa !10
  store i32 %284, ptr %167, align 4, !tbaa !10
  %285 = getelementptr inbounds nuw i8, ptr %166, i64 16520
  %286 = load i32, ptr %285, align 4, !tbaa !10
  store i32 %286, ptr %167, align 4, !tbaa !10
  %287 = getelementptr inbounds nuw i8, ptr %166, i64 16800
  %288 = load i32, ptr %287, align 4, !tbaa !10
  store i32 %288, ptr %167, align 4, !tbaa !10
  %289 = getelementptr inbounds nuw i8, ptr %166, i64 17080
  %290 = load i32, ptr %289, align 4, !tbaa !10
  store i32 %290, ptr %167, align 4, !tbaa !10
  %291 = getelementptr inbounds nuw i8, ptr %166, i64 17360
  %292 = load i32, ptr %291, align 4, !tbaa !10
  store i32 %292, ptr %167, align 4, !tbaa !10
  %293 = getelementptr inbounds nuw i8, ptr %166, i64 17640
  %294 = load i32, ptr %293, align 4, !tbaa !10
  store i32 %294, ptr %167, align 4, !tbaa !10
  %295 = getelementptr inbounds nuw i8, ptr %166, i64 17920
  %296 = load i32, ptr %295, align 4, !tbaa !10
  store i32 %296, ptr %167, align 4, !tbaa !10
  %297 = getelementptr inbounds nuw i8, ptr %166, i64 18200
  %298 = load i32, ptr %297, align 4, !tbaa !10
  store i32 %298, ptr %167, align 4, !tbaa !10
  %299 = getelementptr inbounds nuw i8, ptr %166, i64 18480
  %300 = load i32, ptr %299, align 4, !tbaa !10
  store i32 %300, ptr %167, align 4, !tbaa !10
  %301 = getelementptr inbounds nuw i8, ptr %166, i64 18760
  %302 = load i32, ptr %301, align 4, !tbaa !10
  store i32 %302, ptr %167, align 4, !tbaa !10
  %303 = getelementptr inbounds nuw i8, ptr %166, i64 19040
  %304 = load i32, ptr %303, align 4, !tbaa !10
  store i32 %304, ptr %167, align 4, !tbaa !10
  %305 = getelementptr inbounds nuw i8, ptr %166, i64 19320
  %306 = load i32, ptr %305, align 4, !tbaa !10
  store i32 %306, ptr %167, align 4, !tbaa !10
  %307 = add nuw nsw i64 %165, 1
  %308 = icmp eq i64 %307, 70
  br i1 %308, label %161, label %164, !llvm.loop !12
}

; Function Attrs: nofree nounwind uwtable
define dso_local noundef i32 @main() local_unnamed_addr #2 {
  store i32 5, ptr @b, align 4, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr @c, align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 16), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 32), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 48), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 64), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 80), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 96), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 112), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 128), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 144), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 160), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 176), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 192), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 208), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 224), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 240), align 16, !tbaa !10
  store <4 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 256), align 16, !tbaa !10
  store <2 x i32> splat (i32 2075593088), ptr getelementptr inbounds nuw (i8, ptr @c, i64 272), align 16, !tbaa !10
  tail call void @fn2()
  %1 = load i32, ptr @e, align 4, !tbaa !10
  tail call void @f(ptr noundef nonnull @a, i32 noundef %1)
  %2 = load i64, ptr @a, align 8, !tbaa !6
  %3 = icmp eq i64 %2, 0
  br i1 %3, label %5, label %4

4:                                                ; preds = %0
  tail call void @abort() #5
  unreachable

5:                                                ; preds = %0
  ret i32 0
}

; Function Attrs: cold nofree noreturn nounwind
declare void @abort() local_unnamed_addr #3

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr writeonly captures(none), i8, i64, i1 immarg) #4

attributes #0 = { mustprogress nofree noinline norecurse nosync nounwind willreturn memory(argmem: write) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #1 = { nofree noinline norecurse nosync nounwind memory(readwrite, argmem: none, inaccessiblemem: none) uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #2 = { nofree nounwind uwtable "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #3 = { cold nofree noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+fp-armv8,+neon,+outline-atomics,+v8a,-fmv" }
attributes #4 = { nocallback nofree nounwind willreturn memory(argmem: write) }
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
!7 = !{!"long long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11, !11, i64 0}
!11 = !{!"int", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = distinct !{!14, !13}
