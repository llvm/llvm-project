; RUN: llc -mtriple=hexagon < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that we use the correct name in an epilog phi for a phi value
; that is defined for the last time in the kernel. Previously, we
; used the value from kernel loop definition, but we really need
; to use the value from the Phi in the kernel instead.

; In this test case, the second loop is pipelined, block b5.

; CHECK: loop1
; CHECK: [[REG0:r([0-9]+)]] += mpyi
; CHECK: [[REG2:r([0-9]+)]] = add([[REG1:r([0-9]+)]],add([[REG0]],#8
; CHECK: endloop1

%s.0 = type { ptr, ptr, ptr, ptr, i8, i32, ptr, i32, i32, i32, i8, i8, i32, i32, double, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, i8, i32, i32, i32, i32, i32, i32, ptr, i32, i32, i32, i32, i32, ptr, [4 x ptr], [4 x ptr], [4 x ptr], i32, ptr, i8, i8, [16 x i8], [16 x i8], [16 x i8], i32, i8, i8, i8, i8, i16, i16, i8, i8, i8, ptr, i32, i32, i32, i32, ptr, i32, [4 x ptr], i32, i32, i32, [10 x i32], i32, i32, i32, i32, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }
%s.1 = type { ptr, ptr, ptr, ptr, ptr, i32, %s.3, i32, i32, ptr, i32, ptr, i32, i32 }
%s.2 = type { ptr, ptr, ptr, ptr, i8, i32 }
%s.3 = type { [8 x i32], [48 x i8] }
%s.4 = type { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i32, i32 }
%s.5 = type opaque
%s.6 = type opaque
%s.7 = type { ptr, i32, i32, i32, i32 }
%s.8 = type { ptr, i32, ptr, ptr, ptr, ptr, ptr }
%s.9 = type { [64 x i16], i8 }
%s.10 = type { [17 x i8], [256 x i8], i8 }
%s.11 = type { ptr, i8, i32, i32, ptr }
%s.12 = type { ptr, ptr, i8 }
%s.13 = type { ptr, ptr }
%s.14 = type { ptr, ptr, ptr, ptr, ptr }
%s.15 = type { ptr, ptr }
%s.16 = type { ptr, ptr, ptr, ptr, i8, i8 }
%s.17 = type { ptr, ptr, ptr, i8, i8, i32, i32 }
%s.18 = type { ptr, ptr, i8 }
%s.19 = type { ptr, [5 x ptr] }
%s.20 = type { ptr, ptr, i8 }
%s.21 = type { ptr, ptr }
%s.22 = type { ptr, ptr, ptr, ptr }
%s.23 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, i32, ptr, ptr }

; Function Attrs: nounwind optsize
define hidden void @f0(ptr nocapture readonly %a0, ptr nocapture readonly %a1, ptr nocapture readonly %a2, ptr nocapture readonly %a3) #0 {
b0:
  %v0 = load ptr, ptr %a3, align 4
  %v1 = getelementptr inbounds %s.0, ptr %a0, i32 0, i32 62
  %v2 = load i32, ptr %v1, align 4
  %v3 = icmp sgt i32 %v2, 0
  br i1 %v3, label %b1, label %b10

b1:                                               ; preds = %b0
  %v4 = getelementptr inbounds %s.23, ptr %a1, i32 0, i32 10
  br label %b2

b2:                                               ; preds = %b8, %b1
  %v5 = phi i32 [ 0, %b1 ], [ %v98, %b8 ]
  %v6 = phi i32 [ 0, %b1 ], [ %v99, %b8 ]
  %v7 = getelementptr inbounds ptr, ptr %a2, i32 %v6
  br label %b3

b3:                                               ; preds = %b7, %b2
  %v8 = phi i32 [ 0, %b2 ], [ %v96, %b7 ]
  %v9 = phi i32 [ %v5, %b2 ], [ %v16, %b7 ]
  %v10 = load ptr, ptr %v7, align 4
  %v11 = icmp eq i32 %v8, 0
  %v12 = select i1 %v11, i32 -1, i32 1
  %v13 = add i32 %v12, %v6
  %v14 = getelementptr inbounds ptr, ptr %a2, i32 %v13
  %v15 = load ptr, ptr %v14, align 4
  %v16 = add nsw i32 %v9, 1
  %v17 = getelementptr inbounds ptr, ptr %v0, i32 %v9
  %v18 = load ptr, ptr %v17, align 4
  %v19 = getelementptr inbounds i8, ptr %v10, i32 1
  %v20 = load i8, ptr %v10, align 1
  %v21 = zext i8 %v20 to i32
  %v22 = mul nsw i32 %v21, 3
  %v23 = getelementptr inbounds i8, ptr %v15, i32 1
  %v24 = load i8, ptr %v15, align 1
  %v25 = zext i8 %v24 to i32
  %v26 = add nsw i32 %v22, %v25
  %v27 = load i8, ptr %v19, align 1
  %v28 = zext i8 %v27 to i32
  %v29 = mul nsw i32 %v28, 3
  %v30 = load i8, ptr %v23, align 1
  %v31 = zext i8 %v30 to i32
  %v32 = add nsw i32 %v29, %v31
  %v33 = mul nsw i32 %v26, 4
  %v34 = add nsw i32 %v33, 8
  %v35 = lshr i32 %v34, 4
  %v36 = trunc i32 %v35 to i8
  %v37 = getelementptr inbounds i8, ptr %v18, i32 1
  store i8 %v36, ptr %v18, align 1
  %v38 = mul nsw i32 %v26, 3
  %v39 = add i32 %v38, 7
  %v40 = add i32 %v39, %v32
  %v41 = lshr i32 %v40, 4
  %v42 = trunc i32 %v41 to i8
  store i8 %v42, ptr %v37, align 1
  %v43 = load i32, ptr %v4, align 4
  %v44 = add i32 %v43, -2
  %v45 = getelementptr inbounds i8, ptr %v18, i32 2
  %v46 = icmp eq i32 %v44, 0
  br i1 %v46, label %b7, label %b4

b4:                                               ; preds = %b3
  %v47 = getelementptr inbounds i8, ptr %v15, i32 2
  %v48 = getelementptr inbounds i8, ptr %v10, i32 2
  %v49 = mul i32 %v43, 2
  br label %b5

b5:                                               ; preds = %b5, %b4
  %v50 = phi ptr [ %v45, %b4 ], [ %v76, %b5 ]
  %v51 = phi i32 [ %v44, %b4 ], [ %v75, %b5 ]
  %v52 = phi i32 [ %v26, %b4 ], [ %v53, %b5 ]
  %v53 = phi i32 [ %v32, %b4 ], [ %v64, %b5 ]
  %v54 = phi ptr [ %v18, %b4 ], [ %v50, %b5 ]
  %v55 = phi ptr [ %v47, %b4 ], [ %v61, %b5 ]
  %v56 = phi ptr [ %v48, %b4 ], [ %v57, %b5 ]
  %v57 = getelementptr inbounds i8, ptr %v56, i32 1
  %v58 = load i8, ptr %v56, align 1
  %v59 = zext i8 %v58 to i32
  %v60 = mul nsw i32 %v59, 3
  %v61 = getelementptr inbounds i8, ptr %v55, i32 1
  %v62 = load i8, ptr %v55, align 1
  %v63 = zext i8 %v62 to i32
  %v64 = add nsw i32 %v60, %v63
  %v65 = mul nsw i32 %v53, 3
  %v66 = add i32 %v52, 8
  %v67 = add i32 %v66, %v65
  %v68 = lshr i32 %v67, 4
  %v69 = trunc i32 %v68 to i8
  %v70 = getelementptr inbounds i8, ptr %v54, i32 3
  store i8 %v69, ptr %v50, align 1
  %v71 = add i32 %v65, 7
  %v72 = add i32 %v71, %v64
  %v73 = lshr i32 %v72, 4
  %v74 = trunc i32 %v73 to i8
  store i8 %v74, ptr %v70, align 1
  %v75 = add i32 %v51, -1
  %v76 = getelementptr inbounds i8, ptr %v50, i32 2
  %v77 = icmp eq i32 %v75, 0
  br i1 %v77, label %b6, label %b5

b6:                                               ; preds = %b5
  %v78 = add i32 %v49, -2
  %v79 = getelementptr i8, ptr %v18, i32 %v78
  %v80 = add i32 %v49, -4
  %v81 = getelementptr i8, ptr %v18, i32 %v80
  br label %b7

b7:                                               ; preds = %b6, %b3
  %v82 = phi ptr [ %v79, %b6 ], [ %v45, %b3 ]
  %v83 = phi i32 [ %v53, %b6 ], [ %v26, %b3 ]
  %v84 = phi i32 [ %v64, %b6 ], [ %v32, %b3 ]
  %v85 = phi ptr [ %v81, %b6 ], [ %v18, %b3 ]
  %v86 = mul nsw i32 %v84, 3
  %v87 = add i32 %v83, 8
  %v88 = add i32 %v87, %v86
  %v89 = lshr i32 %v88, 4
  %v90 = trunc i32 %v89 to i8
  %v91 = getelementptr inbounds i8, ptr %v85, i32 3
  store i8 %v90, ptr %v82, align 1
  %v92 = mul nsw i32 %v84, 4
  %v93 = add nsw i32 %v92, 7
  %v94 = lshr i32 %v93, 4
  %v95 = trunc i32 %v94 to i8
  store i8 %v95, ptr %v91, align 1
  %v96 = add nsw i32 %v8, 1
  %v97 = icmp eq i32 %v96, 2
  br i1 %v97, label %b8, label %b3

b8:                                               ; preds = %b7
  %v98 = add i32 %v5, 2
  %v99 = add nsw i32 %v6, 1
  %v100 = load i32, ptr %v1, align 4
  %v101 = icmp slt i32 %v98, %v100
  br i1 %v101, label %b2, label %b9

b9:                                               ; preds = %b8
  br label %b10

b10:                                              ; preds = %b9, %b0
  ret void
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv60" }
