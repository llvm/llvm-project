; RUN: opt < %s -passes=loop-interchange --pass-remarks-missed=loop-interchange -disable-output 2>&1 | FileCheck %s
; RUN: opt < %s -passes=loop-interchange --pass-remarks-missed=loop-interchange -loop-interchange-max-meminstr-count=75
;               -disable-output 2>&1 | FileCheck --check-prefix=CHECK-INSTR-COUNT %s
target triple = "aarch64-unknown-linux-gnu"

@A = dso_local local_unnamed_addr global [2048 x [2048 x i32]] zeroinitializer, align 4
@B = dso_local local_unnamed_addr global [2048 x [2048 x i32]] zeroinitializer, align 4
@C = dso_local local_unnamed_addr global [2048 x [2048 x i32]] zeroinitializer, align 4

; CHECK: Number of loads/stores exceeded, the supported maximum
;        can be increased with option -loop-interchange-maxmeminstr-count.
; CHECK-INSTR-COUNT-NOT: Number of loads/stores exceeded, the supported maximum
;        can be increased with option -loop-interchange-maxmeminstr-count.
define dso_local noundef i32 @many_load_stores() {
  br label %1

1:                                                ; preds = %9, %0
  %2 = phi i32 [ 0, %0 ], [ %10, %9 ]
  %3 = icmp slt i32 %2, 2048
  br i1 %3, label %5, label %4

4:                                                ; preds = %1
  ret i32 0

5:                                                ; preds = %1
  br label %6

6:                                                ; preds = %11, %5
  %7 = phi i32 [ 0, %5 ], [ %208, %11 ]
  %8 = icmp slt i32 %7, 85
  br i1 %8, label %11, label %9

9:                                                ; preds = %6
  %10 = add nsw i32 %2, 1
  br label %1

11:                                               ; preds = %6
  %12 = sext i32 %2 to i64
  %13 = getelementptr inbounds [2048 x [2048 x i32]], [2048 x [2048 x i32]]* @B, i64 0, i64 %12
  %14 = sext i32 %7 to i64
  %15 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %14
  %16 = load i32, i32* %15, align 4
  %17 = getelementptr inbounds [2048 x [2048 x i32]], [2048 x [2048 x i32]]* @C, i64 0, i64 %12
  %18 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %14
  %19 = load i32, i32* %18, align 4
  %20 = add nsw i32 %16, %19
  %21 = getelementptr inbounds [2048 x [2048 x i32]], [2048 x [2048 x i32]]* @A, i64 0, i64 %12
  %22 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %14
  store i32 %20, i32* %22, align 4
  %23 = add nsw i32 %7, 1
  %24 = sext i32 %23 to i64
  %25 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %24
  %26 = load i32, i32* %25, align 4
  %27 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %24
  %28 = load i32, i32* %27, align 4
  %29 = add nsw i32 %26, %28
  %30 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %24
  store i32 %29, i32* %30, align 4
  %31 = add nsw i32 %23, 1
  %32 = sext i32 %31 to i64
  %33 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %32
  %34 = load i32, i32* %33, align 4
  %35 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %32
  %36 = load i32, i32* %35, align 4
  %37 = add nsw i32 %34, %36
  %38 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %32
  store i32 %37, i32* %38, align 4
  %39 = add nsw i32 %31, 1
  %40 = sext i32 %39 to i64
  %41 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %40
  %42 = load i32, i32* %41, align 4
  %43 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %40
  %44 = load i32, i32* %43, align 4
  %45 = add nsw i32 %42, %44
  %46 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %40
  store i32 %45, i32* %46, align 4
  %47 = add nsw i32 %39, 1
  %48 = sext i32 %47 to i64
  %49 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %48
  %50 = load i32, i32* %49, align 4
  %51 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %48
  %52 = load i32, i32* %51, align 4
  %53 = add nsw i32 %50, %52
  %54 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %48
  store i32 %53, i32* %54, align 4
  %55 = add nsw i32 %47, 1
  %56 = sext i32 %55 to i64
  %57 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %56
  %58 = load i32, i32* %57, align 4
  %59 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %56
  %60 = load i32, i32* %59, align 4
  %61 = add nsw i32 %58, %60
  %62 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %56
  store i32 %61, i32* %62, align 4
  %63 = add nsw i32 %55, 1
  %64 = sext i32 %63 to i64
  %65 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %64
  %66 = load i32, i32* %65, align 4
  %67 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %64
  %68 = load i32, i32* %67, align 4
  %69 = add nsw i32 %66, %68
  %70 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %64
  store i32 %69, i32* %70, align 4
  %71 = add nsw i32 %63, 1
  %72 = sext i32 %71 to i64
  %73 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %72
  %74 = load i32, i32* %73, align 4
  %75 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %72
  %76 = load i32, i32* %75, align 4
  %77 = add nsw i32 %74, %76
  %78 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %72
  store i32 %77, i32* %78, align 4
  %79 = add nsw i32 %71, 1
  %80 = sext i32 %79 to i64
  %81 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %80
  %82 = load i32, i32* %81, align 4
  %83 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %80
  %84 = load i32, i32* %83, align 4
  %85 = add nsw i32 %82, %84
  %86 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %80
  store i32 %85, i32* %86, align 4
  %87 = add nsw i32 %79, 1
  %88 = sext i32 %87 to i64
  %89 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %88
  %90 = load i32, i32* %89, align 4
  %91 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %88
  %92 = load i32, i32* %91, align 4
  %93 = add nsw i32 %90, %92
  %94 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %88
  store i32 %93, i32* %94, align 4
  %95 = add nsw i32 %87, 1
  %96 = sext i32 %95 to i64
  %97 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %96
  %98 = load i32, i32* %97, align 4
  %99 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %96
  %100 = load i32, i32* %99, align 4
  %101 = add nsw i32 %98, %100
  %102 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %96
  store i32 %101, i32* %102, align 4
  %103 = add nsw i32 %95, 1
  %104 = sext i32 %103 to i64
  %105 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %104
  %106 = load i32, i32* %105, align 4
  %107 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %104
  %108 = load i32, i32* %107, align 4
  %109 = add nsw i32 %106, %108
  %110 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %104
  store i32 %109, i32* %110, align 4
  %111 = add nsw i32 %103, 1
  %112 = sext i32 %111 to i64
  %113 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %112
  %114 = load i32, i32* %113, align 4
  %115 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %112
  %116 = load i32, i32* %115, align 4
  %117 = add nsw i32 %114, %116
  %118 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %112
  store i32 %117, i32* %118, align 4
  %119 = add nsw i32 %111, 1
  %120 = sext i32 %119 to i64
  %121 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %120
  %122 = load i32, i32* %121, align 4
  %123 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %120
  %124 = load i32, i32* %123, align 4
  %125 = add nsw i32 %122, %124
  %126 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %120
  store i32 %125, i32* %126, align 4
  %127 = add nsw i32 %119, 1
  %128 = sext i32 %127 to i64
  %129 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %128
  %130 = load i32, i32* %129, align 4
  %131 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %128
  %132 = load i32, i32* %131, align 4
  %133 = add nsw i32 %130, %132
  %134 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %128
  store i32 %133, i32* %134, align 4
  %135 = add nsw i32 %127, 1
  %136 = sext i32 %135 to i64
  %137 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %136
  %138 = load i32, i32* %137, align 4
  %139 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %136
  %140 = load i32, i32* %139, align 4
  %141 = add nsw i32 %138, %140
  %142 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %136
  store i32 %141, i32* %142, align 4
  %143 = add nsw i32 %135, 1
  %144 = sext i32 %143 to i64
  %145 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %144
  %146 = load i32, i32* %145, align 4
  %147 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %144
  %148 = load i32, i32* %147, align 4
  %149 = add nsw i32 %146, %148
  %150 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %144
  store i32 %149, i32* %150, align 4
  %151 = add nsw i32 %143, 1
  %152 = sext i32 %151 to i64
  %153 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %152
  %154 = load i32, i32* %153, align 4
  %155 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %152
  %156 = load i32, i32* %155, align 4
  %157 = add nsw i32 %154, %156
  %158 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %152
  store i32 %157, i32* %158, align 4
  %159 = add nsw i32 %151, 1
  %160 = sext i32 %159 to i64
  %161 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %160
  %162 = load i32, i32* %161, align 4
  %163 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %160
  %164 = load i32, i32* %163, align 4
  %165 = add nsw i32 %162, %164
  %166 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %160
  store i32 %165, i32* %166, align 4
  %167 = add nsw i32 %159, 1
  %168 = sext i32 %167 to i64
  %169 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %168
  %170 = load i32, i32* %169, align 4
  %171 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %168
  %172 = load i32, i32* %171, align 4
  %173 = add nsw i32 %170, %172
  %174 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %168
  store i32 %173, i32* %174, align 4
  %175 = add nsw i32 %167, 1
  %176 = sext i32 %175 to i64
  %177 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %176
  %178 = load i32, i32* %177, align 4
  %179 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %176
  %180 = load i32, i32* %179, align 4
  %181 = add nsw i32 %178, %180
  %182 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %176
  store i32 %181, i32* %182, align 4
  %183 = add nsw i32 %175, 1
  %184 = sext i32 %183 to i64
  %185 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %184
  %186 = load i32, i32* %185, align 4
  %187 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %184
  %188 = load i32, i32* %187, align 4
  %189 = add nsw i32 %186, %188
  %190 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %184
  store i32 %189, i32* %190, align 4
  %191 = add nsw i32 %183, 1
  %192 = sext i32 %191 to i64
  %193 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %192
  %194 = load i32, i32* %193, align 4
  %195 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %192
  %196 = load i32, i32* %195, align 4
  %197 = add nsw i32 %194, %196
  %198 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %192
  store i32 %197, i32* %198, align 4
  %199 = add nsw i32 %191, 1
  %200 = sext i32 %199 to i64
  %201 = getelementptr inbounds [2048 x i32], [2048 x i32]* %13, i64 0, i64 %200
  %202 = load i32, i32* %201, align 4
  %203 = getelementptr inbounds [2048 x i32], [2048 x i32]* %17, i64 0, i64 %200
  %204 = load i32, i32* %203, align 4
  %205 = add nsw i32 %202, %204
  %206 = getelementptr inbounds [2048 x i32], [2048 x i32]* %21, i64 0, i64 %200
  store i32 %205, i32* %206, align 4
  %207 = add nsw i32 %199, 1
  %208 = add nsw i32 %207, 24
  br label %6
}