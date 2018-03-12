; RUN: llc -march=hexagon -O0 < %s | FileCheck --check-prefix=CHECKO0 %s
; KP: Removed -O2 check. The code has become more aggressively optimized
; (some loads were found to be redundant and have been removed completely),
; and verifying correct code generation has become more difficult than
; its worth.

; CHECK: v{{[0-9]*}} = vsplat(r{{[0-9]*}})
; CHECK: v{{[0-9]*}} = vsplat(r{{[0-9]*}})

; CHECKO0: vmem(r{{[0-9]*}}+#0) = v{{[0-9]*}}
; CHECKO0: v{{[0-9]*}} = vmem(r{{[0-9]*}}+#0)
; CHECKO0: v{{[0-9]*}} = vmem(r{{[0-9]*}}+#0)

; Allow .cur loads.
; CHECKO2: v{{[0-9].*}} = vmem(r{{[0-9]*}}+#0)
; CHECKO2: vmem(r{{[0-9]*}}+#0) = v{{[0-9]*}}
; CHECKO2: v{{[0-9].*}} = vmem(r{{[0-9]*}}+#0)

; CHECK: v{{[0-9]*}}:{{[0-9]*}} = vcombine(v{{[0-9]*}},v{{[0-9]*}})
; CHECK: vmem(r{{[0-9]*}}+#0) = v{{[0-9]*}}
; CHECK: vmem(r{{[0-9]*}}+#32) = v{{[0-9]*}}
; CHECK: v{{[0-9]*}} = vmem(r{{[0-9]*}}+#0)
; CHECK: v{{[0-9]*}} = vmem(r{{[0-9]*}}+#32)
; CHECK: vmem(r{{[0-9]*}}+#0) = v{{[0-9]*}}
; CHECK: vmem(r{{[0-9]*}}+#32) = v{{[0-9]*}}

target triple = "hexagon"

@g0 = common global [10 x <32 x i32>] zeroinitializer, align 64
@g1 = private unnamed_addr constant [11 x i8] c"c[%d]= %x\0A\00", align 8
@g2 = common global [10 x <16 x i32>] zeroinitializer, align 64
@g3 = common global [10 x <16 x i32>] zeroinitializer, align 64
@g4 = common global [10 x <32 x i32>] zeroinitializer, align 64

declare i32 @f0(i8*, ...)

; Function Attrs: nounwind
define void @f1(i32 %a0) #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32*, align 4
  %v2 = alloca i32, align 4
  store i32 %a0, i32* %v0, align 4
  store i32* getelementptr inbounds ([10 x <32 x i32>], [10 x <32 x i32>]* @g0, i32 0, i32 0, i32 0), i32** %v1, align 4
  %v3 = load i32, i32* %v0, align 4
  %v4 = load i32*, i32** %v1, align 4
  %v5 = getelementptr inbounds i32, i32* %v4, i32 %v3
  store i32* %v5, i32** %v1, align 4
  store i32 0, i32* %v2, align 4
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v6 = load i32, i32* %v2, align 4
  %v7 = icmp slt i32 %v6, 16
  br i1 %v7, label %b2, label %b4

b2:                                               ; preds = %b1
  %v8 = load i32, i32* %v2, align 4
  %v9 = load i32*, i32** %v1, align 4
  %v10 = getelementptr inbounds i32, i32* %v9, i32 1
  store i32* %v10, i32** %v1, align 4
  %v11 = load i32, i32* %v9, align 4
  %v12 = call i32 (i8*, ...) @f0(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @g1, i32 0, i32 0), i32 %v8, i32 %v11)
  br label %b3

b3:                                               ; preds = %b2
  %v13 = load i32, i32* %v2, align 4
  %v14 = add nsw i32 %v13, 1
  store i32 %v14, i32* %v2, align 4
  br label %b1

b4:                                               ; preds = %b1
  ret void
}

; Function Attrs: nounwind
define i32 @f2() #0 {
b0:
  %v0 = alloca i32, align 4
  %v1 = alloca i32, align 4
  store i32 0, i32* %v0
  store i32 0, i32* %v1, align 4
  br label %b1

b1:                                               ; preds = %b3, %b0
  %v2 = load i32, i32* %v1, align 4
  %v3 = icmp slt i32 %v2, 3
  br i1 %v3, label %b2, label %b4

b2:                                               ; preds = %b1
  %v4 = load i32, i32* %v1, align 4
  %v5 = add nsw i32 %v4, 1
  %v6 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v5)
  %v7 = load i32, i32* %v1, align 4
  %v8 = getelementptr inbounds [10 x <16 x i32>], [10 x <16 x i32>]* @g2, i32 0, i32 %v7
  store <16 x i32> %v6, <16 x i32>* %v8, align 64
  %v9 = load i32, i32* %v1, align 4
  %v10 = mul nsw i32 %v9, 10
  %v11 = add nsw i32 %v10, 1
  %v12 = call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 %v11)
  %v13 = load i32, i32* %v1, align 4
  %v14 = getelementptr inbounds [10 x <16 x i32>], [10 x <16 x i32>]* @g3, i32 0, i32 %v13
  store <16 x i32> %v12, <16 x i32>* %v14, align 64
  %v15 = load i32, i32* %v1, align 4
  %v16 = getelementptr inbounds [10 x <16 x i32>], [10 x <16 x i32>]* @g2, i32 0, i32 %v15
  %v17 = load <16 x i32>, <16 x i32>* %v16, align 64
  %v18 = load i32, i32* %v1, align 4
  %v19 = getelementptr inbounds [10 x <16 x i32>], [10 x <16 x i32>]* @g3, i32 0, i32 %v18
  %v20 = load <16 x i32>, <16 x i32>* %v19, align 64
  %v21 = call <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32> %v17, <16 x i32> %v20)
  %v22 = load i32, i32* %v1, align 4
  %v23 = getelementptr inbounds [10 x <32 x i32>], [10 x <32 x i32>]* @g4, i32 0, i32 %v22
  store <32 x i32> %v21, <32 x i32>* %v23, align 64
  br label %b3

b3:                                               ; preds = %b2
  %v24 = load i32, i32* %v1, align 4
  %v25 = add nsw i32 %v24, 1
  store i32 %v25, i32* %v1, align 4
  br label %b1

b4:                                               ; preds = %b1
  store i32 0, i32* %v1, align 4
  br label %b5

b5:                                               ; preds = %b7, %b4
  %v26 = load i32, i32* %v1, align 4
  %v27 = icmp slt i32 %v26, 3
  br i1 %v27, label %b6, label %b8

b6:                                               ; preds = %b5
  %v28 = load i32, i32* %v1, align 4
  %v29 = getelementptr inbounds [10 x <32 x i32>], [10 x <32 x i32>]* @g4, i32 0, i32 %v28
  %v30 = load <32 x i32>, <32 x i32>* %v29, align 64
  %v31 = load i32, i32* %v1, align 4
  %v32 = getelementptr inbounds [10 x <32 x i32>], [10 x <32 x i32>]* @g0, i32 0, i32 %v31
  store <32 x i32> %v30, <32 x i32>* %v32, align 64
  br label %b7

b7:                                               ; preds = %b6
  %v33 = load i32, i32* %v1, align 4
  %v34 = add nsw i32 %v33, 1
  store i32 %v34, i32* %v1, align 4
  br label %b5

b8:                                               ; preds = %b5
  store i32 0, i32* %v1, align 4
  br label %b9

b9:                                               ; preds = %b11, %b8
  %v35 = load i32, i32* %v1, align 4
  %v36 = icmp slt i32 %v35, 3
  br i1 %v36, label %b10, label %b12

b10:                                              ; preds = %b9
  %v37 = load i32, i32* %v1, align 4
  %v38 = mul nsw i32 %v37, 16
  call void @f1(i32 %v38)
  br label %b11

b11:                                              ; preds = %b10
  %v39 = load i32, i32* %v1, align 4
  %v40 = add nsw i32 %v39, 1
  store i32 %v40, i32* %v1, align 4
  br label %b9

b12:                                              ; preds = %b9
  ret i32 0
}

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32) #1

; Function Attrs: nounwind readnone
declare <32 x i32> @llvm.hexagon.V6.vcombine(<16 x i32>, <16 x i32>) #1

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
attributes #1 = { nounwind readnone }
