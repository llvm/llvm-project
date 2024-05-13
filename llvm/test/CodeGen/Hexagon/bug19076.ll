; REQUIRES: asserts
; RUN: llc -march=hexagon -stats -o /dev/null < %s

%s.0 = type { ptr, ptr, ptr, i32, i32, i32, i32, ptr, ptr, ptr }
%s.1 = type opaque
%s.2 = type { %s.3, ptr, ptr }
%s.3 = type { i32, i32 }
%s.4 = type { ptr, ptr, ptr, ptr, i32, i32, i32, ptr, i32, ptr, ptr, i8, i8, i8, ptr, i32, ptr, ptr, i8, i8, i8, ptr, i32, ptr, i32, ptr, i8, %s.9, [32 x ptr], [7 x ptr], i32, ptr, i32, ptr, i32, i32, %s.11, %s.13, i8, i8, i8, ptr, ptr, ptr, i32, [12 x i8] }
%s.5 = type { [1 x %s.6], i32, %s.7, [4 x i8] }
%s.6 = type { [16 x i32] }
%s.7 = type { [2 x i32] }
%s.8 = type { ptr, ptr, i32, ptr }
%s.9 = type { ptr, ptr, %s.7, i32, %s.10 }
%s.10 = type { i32 }
%s.11 = type { %s.12, i8, ptr }
%s.12 = type { [2 x i32] }
%s.13 = type { i32, i32 }
%s.14 = type { ptr, ptr }
%s.15 = type { ptr, ptr, i32 }
%s.16 = type { %s.3, i32, ptr, ptr, ptr, i32, i32 }
%s.17 = type { i32, ptr }
%s.18 = type { ptr, ptr }

; Function Attrs: nounwind
define zeroext i8 @f0(ptr %a0, i32 %a1, ptr %a2) #0 {
b0:
  %v0 = alloca i8, align 1
  %v1 = alloca ptr, align 4
  %v2 = alloca i32, align 4
  %v3 = alloca ptr, align 4
  %v4 = alloca i32, align 4
  %v5 = alloca i32, align 4
  %v6 = alloca ptr
  %v7 = alloca i32, align 4
  %v8 = alloca i32
  %v9 = alloca %s.4, align 32
  store ptr %a0, ptr %v1, align 4
  store i32 %a1, ptr %v2, align 4
  store ptr %a2, ptr %v3, align 4
  %v10 = load ptr, ptr %v1, align 4
  %v11 = getelementptr inbounds %s.0, ptr %v10, i32 0, i32 3
  %v12 = load i32, ptr %v11, align 4
  store i32 %v12, ptr %v4, align 4
  %v13 = load ptr, ptr %v1, align 4
  %v14 = getelementptr inbounds %s.0, ptr %v13, i32 0, i32 6
  %v15 = load i32, ptr %v14, align 4
  store i32 %v15, ptr %v5, align 4
  %v16 = load i32, ptr %v4, align 4
  %v17 = call ptr @llvm.stacksave()
  store ptr %v17, ptr %v6
  %v18 = alloca %s.2, i32 %v16, align 8
  %v19 = load ptr, ptr %v1, align 4
  %v20 = call i32 @f1(ptr %v19)
  %v21 = icmp ne i32 %v20, 0
  br i1 %v21, label %b2, label %b1

b1:                                               ; preds = %b0
  store i8 8, ptr %v0
  store i32 1, ptr %v8
  br label %b23

b2:                                               ; preds = %b0
  %v22 = load ptr, ptr %v1, align 4
  %v24 = load ptr, ptr %v22, align 4
  %v25 = load ptr, ptr %v1, align 4
  %v26 = getelementptr inbounds %s.0, ptr %v25, i32 0, i32 1
  %v27 = load ptr, ptr %v26, align 4
  %v30 = load i32, ptr %v4, align 4
  %v31 = mul i32 16, %v30
  %v32 = call zeroext i8 @f2(ptr %v24, ptr %v27, ptr %v18, i32 %v31)
  %v33 = zext i8 %v32 to i32
  %v34 = icmp ne i32 %v33, 0
  br i1 %v34, label %b3, label %b4

b3:                                               ; preds = %b2
  store i8 1, ptr %v0
  store i32 1, ptr %v8
  br label %b23

b4:                                               ; preds = %b2
  store i32 0, ptr %v7, align 4
  br label %b5

b5:                                               ; preds = %b21, %b4
  %v35 = load i32, ptr %v7, align 4
  %v36 = load i32, ptr %v4, align 4
  %v37 = icmp ult i32 %v35, %v36
  br i1 %v37, label %b6, label %b7

b6:                                               ; preds = %b5
  br label %b7

b7:                                               ; preds = %b6, %b5
  %v38 = phi i1 [ false, %b5 ], [ true, %b6 ]
  br i1 %v38, label %b8, label %b22

b8:                                               ; preds = %b7
  %v39 = load i32, ptr %v7, align 4
  %v40 = getelementptr inbounds %s.2, ptr %v18, i32 %v39
  %v41 = getelementptr inbounds %s.2, ptr %v40, i32 0, i32 1
  %v42 = load ptr, ptr %v41, align 4
  %v43 = icmp ne ptr %v42, null
  br i1 %v43, label %b9, label %b17

b9:                                               ; preds = %b8
  %v44 = load ptr, ptr %v1, align 4
  %v46 = load ptr, ptr %v44, align 4
  %v47 = load i32, ptr %v7, align 4
  %v48 = getelementptr inbounds %s.2, ptr %v18, i32 %v47
  %v49 = getelementptr inbounds %s.2, ptr %v48, i32 0, i32 1
  %v50 = load ptr, ptr %v49, align 4
  %v53 = load i32, ptr %v5, align 4
  %v54 = call zeroext i8 @f2(ptr %v46, ptr %v50, ptr %v9, i32 %v53)
  %v55 = zext i8 %v54 to i32
  %v56 = icmp ne i32 %v55, 0
  br i1 %v56, label %b10, label %b11

b10:                                              ; preds = %b9
  store i8 1, ptr %v0
  store i32 1, ptr %v8
  br label %b23

b11:                                              ; preds = %b9
  %v57 = getelementptr inbounds %s.4, ptr %v9, i32 0, i32 5
  %v58 = load i32, ptr %v57, align 4
  %v59 = icmp ne i32 %v58, 0
  br i1 %v59, label %b12, label %b13

b12:                                              ; preds = %b11
  br label %b14

b13:                                              ; preds = %b11
  %v60 = load ptr, ptr %v1, align 4
  %v62 = load ptr, ptr %v60, align 4
  %v63 = call i32 @f3(ptr %v62)
  br label %b14

b14:                                              ; preds = %b13, %b12
  %v64 = phi i32 [ %v58, %b12 ], [ %v63, %b13 ]
  %v65 = load i32, ptr %v2, align 4
  %v66 = icmp eq i32 %v64, %v65
  br i1 %v66, label %b15, label %b16

b15:                                              ; preds = %b14
  %v67 = load ptr, ptr %v1, align 4
  %v68 = load ptr, ptr %v3, align 4
  store ptr %v67, ptr %v68, align 4
  %v70 = load i32, ptr %v7, align 4
  %v71 = getelementptr inbounds %s.2, ptr %v18, i32 %v70
  %v72 = getelementptr inbounds %s.2, ptr %v71, i32 0, i32 1
  %v73 = load ptr, ptr %v72, align 4
  %v75 = load ptr, ptr %v3, align 4
  %v76 = getelementptr inbounds %s.18, ptr %v75, i32 0, i32 1
  store ptr %v73, ptr %v76, align 4
  store i8 0, ptr %v0
  store i32 1, ptr %v8
  br label %b23

b16:                                              ; preds = %b14
  br label %b20

b17:                                              ; preds = %b8
  %v77 = load i32, ptr %v7, align 4
  %v78 = icmp eq i32 %v77, 0
  br i1 %v78, label %b18, label %b19

b18:                                              ; preds = %b17
  %v79 = load ptr, ptr %v1, align 4
  %v80 = load ptr, ptr %v3, align 4
  store ptr %v79, ptr %v80, align 4
  %v82 = load ptr, ptr %v3, align 4
  %v83 = getelementptr inbounds %s.18, ptr %v82, i32 0, i32 1
  store ptr null, ptr %v83, align 4
  store i8 0, ptr %v0
  store i32 1, ptr %v8
  br label %b23

b19:                                              ; preds = %b17
  br label %b20

b20:                                              ; preds = %b19, %b16
  br label %b21

b21:                                              ; preds = %b20
  %v84 = load i32, ptr %v7, align 4
  %v85 = add i32 %v84, 1
  store i32 %v85, ptr %v7, align 4
  br label %b5

b22:                                              ; preds = %b7
  store i8 4, ptr %v0
  store i32 1, ptr %v8
  br label %b23

b23:                                              ; preds = %b22, %b18, %b15, %b10, %b3, %b1
  %v86 = load ptr, ptr %v6
  call void @llvm.stackrestore(ptr %v86)
  %v87 = load i8, ptr %v0
  ret i8 %v87
}

; Function Attrs: nounwind
declare ptr @llvm.stacksave() #0

; Function Attrs: inlinehint nounwind
declare i32 @f1(ptr) #1

declare zeroext i8 @f2(ptr, ptr, ptr, i32) #0

declare i32 @f3(ptr) #0

; Function Attrs: nounwind
declare void @llvm.stackrestore(ptr) #0

attributes #0 = { nounwind }
attributes #1 = { inlinehint nounwind }
