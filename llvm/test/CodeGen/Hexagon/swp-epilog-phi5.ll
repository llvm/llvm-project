; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that we use the correct name in an epilog phi for a phi value
; that is defined for the last time in the kernel. Previously, we
; used the value from kernel loop definition, but we really need
; to use the value from the Phi in the kernel instead.

; In this test case, the second loop is pipelined, block b5.

; CHECK: loop0
; CHECK: [[REG0:r([0-9]+)]] += mpyi
; CHECK-NOT: r{{[0-9]+}} += add([[REG0]],#8)
; CHECK: endloop1

%s.0 = type { %s.1*, %s.4*, %s.7*, i8*, i8, i32, %s.8*, i32, i32, i32, i8, i8, i32, i32, double, i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, i8, i32, i32, i32, i32, i32, i32, i8**, i32, i32, i32, i32, i32, [64 x i32]*, [4 x %s.9*], [4 x %s.10*], [4 x %s.10*], i32, %s.23*, i8, i8, [16 x i8], [16 x i8], [16 x i8], i32, i8, i8, i8, i8, i16, i16, i8, i8, i8, %s.11*, i32, i32, i32, i32, i8*, i32, [4 x %s.23*], i32, i32, i32, [10 x i32], i32, i32, i32, i32, i32, %s.12*, %s.13*, %s.14*, %s.15*, %s.16*, %s.17*, %s.18*, %s.19*, %s.20*, %s.21*, %s.22* }
%s.1 = type { void (%s.2*)*, void (%s.2*, i32)*, void (%s.2*)*, void (%s.2*, i8*)*, void (%s.2*)*, i32, %s.3, i32, i32, i8**, i32, i8**, i32, i32 }
%s.2 = type { %s.1*, %s.4*, %s.7*, i8*, i8, i32 }
%s.3 = type { [8 x i32], [48 x i8] }
%s.4 = type { i8* (%s.2*, i32, i32)*, i8* (%s.2*, i32, i32)*, i8** (%s.2*, i32, i32, i32)*, [64 x i16]** (%s.2*, i32, i32, i32)*, %s.5* (%s.2*, i32, i8, i32, i32, i32)*, %s.6* (%s.2*, i32, i8, i32, i32, i32)*, {}*, i8** (%s.2*, %s.5*, i32, i32, i8)*, [64 x i16]** (%s.2*, %s.6*, i32, i32, i8)*, void (%s.2*, i32)*, {}*, i32, i32 }
%s.5 = type opaque
%s.6 = type opaque
%s.7 = type { {}*, i32, i32, i32, i32 }
%s.8 = type { i8*, i32, {}*, i8 (%s.0*)*, void (%s.0*, i32)*, i8 (%s.0*, i32)*, {}* }
%s.9 = type { [64 x i16], i8 }
%s.10 = type { [17 x i8], [256 x i8], i8 }
%s.11 = type { %s.11*, i8, i32, i32, i8* }
%s.12 = type { {}*, {}*, i8 }
%s.13 = type { void (%s.0*, i8)*, void (%s.0*, i8**, i32*, i32)* }
%s.14 = type { {}*, i32 (%s.0*)*, {}*, i32 (%s.0*, i8***)*, %s.6** }
%s.15 = type { void (%s.0*, i8)*, void (%s.0*, i8***, i32*, i32, i8**, i32*, i32)* }
%s.16 = type { i32 (%s.0*)*, {}*, {}*, {}*, i8, i8 }
%s.17 = type { {}*, i32 (%s.0*)*, i8 (%s.0*)*, i8, i8, i32, i32 }
%s.18 = type { {}*, i8 (%s.0*, [64 x i16]**)*, i8 }
%s.19 = type { {}*, [5 x void (%s.0*, %s.23*, i16*, i8**, i32)*] }
%s.20 = type { {}*, void (%s.0*, i8***, i32*, i32, i8**, i32*, i32)*, i8 }
%s.21 = type { {}*, void (%s.0*, i8***, i32, i8**, i32)* }
%s.22 = type { void (%s.0*, i8)*, void (%s.0*, i8**, i8**, i32)*, {}*, {}* }
%s.23 = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i8, i32, i32, i32, i32, i32, i32, %s.9*, i8* }

; Function Attrs: nounwind optsize
define hidden void @f0(%s.0* nocapture readonly %a0, %s.23* nocapture readonly %a1, i8** nocapture readonly %a2, i8*** nocapture readonly %a3) #0 {
b0:
  %v0 = load i8**, i8*** %a3, align 4
  %v1 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 62
  %v2 = load i32, i32* %v1, align 4
  %v3 = icmp sgt i32 %v2, 0
  br i1 %v3, label %b1, label %b10

b1:                                               ; preds = %b0
  %v4 = getelementptr inbounds %s.23, %s.23* %a1, i32 0, i32 10
  br label %b2

b2:                                               ; preds = %b8, %b1
  %v5 = phi i32 [ 0, %b1 ], [ %v98, %b8 ]
  %v6 = phi i32 [ 0, %b1 ], [ %v99, %b8 ]
  %v7 = getelementptr inbounds i8*, i8** %a2, i32 %v6
  br label %b3

b3:                                               ; preds = %b7, %b2
  %v8 = phi i32 [ 0, %b2 ], [ %v96, %b7 ]
  %v9 = phi i32 [ %v5, %b2 ], [ %v16, %b7 ]
  %v10 = load i8*, i8** %v7, align 4
  %v11 = icmp eq i32 %v8, 0
  %v12 = select i1 %v11, i32 -1, i32 1
  %v13 = add i32 %v12, %v6
  %v14 = getelementptr inbounds i8*, i8** %a2, i32 %v13
  %v15 = load i8*, i8** %v14, align 4
  %v16 = add nsw i32 %v9, 1
  %v17 = getelementptr inbounds i8*, i8** %v0, i32 %v9
  %v18 = load i8*, i8** %v17, align 4
  %v19 = getelementptr inbounds i8, i8* %v10, i32 1
  %v20 = load i8, i8* %v10, align 1
  %v21 = zext i8 %v20 to i32
  %v22 = mul nsw i32 %v21, 3
  %v23 = getelementptr inbounds i8, i8* %v15, i32 1
  %v24 = load i8, i8* %v15, align 1
  %v25 = zext i8 %v24 to i32
  %v26 = add nsw i32 %v22, %v25
  %v27 = load i8, i8* %v19, align 1
  %v28 = zext i8 %v27 to i32
  %v29 = mul nsw i32 %v28, 3
  %v30 = load i8, i8* %v23, align 1
  %v31 = zext i8 %v30 to i32
  %v32 = add nsw i32 %v29, %v31
  %v33 = mul nsw i32 %v26, 4
  %v34 = add nsw i32 %v33, 8
  %v35 = lshr i32 %v34, 4
  %v36 = trunc i32 %v35 to i8
  %v37 = getelementptr inbounds i8, i8* %v18, i32 1
  store i8 %v36, i8* %v18, align 1
  %v38 = mul nsw i32 %v26, 3
  %v39 = add i32 %v38, 7
  %v40 = add i32 %v39, %v32
  %v41 = lshr i32 %v40, 4
  %v42 = trunc i32 %v41 to i8
  store i8 %v42, i8* %v37, align 1
  %v43 = load i32, i32* %v4, align 4
  %v44 = add i32 %v43, -2
  %v45 = getelementptr inbounds i8, i8* %v18, i32 2
  %v46 = icmp eq i32 %v44, 0
  br i1 %v46, label %b7, label %b4

b4:                                               ; preds = %b3
  %v47 = getelementptr inbounds i8, i8* %v15, i32 2
  %v48 = getelementptr inbounds i8, i8* %v10, i32 2
  %v49 = mul i32 %v43, 2
  br label %b5

b5:                                               ; preds = %b5, %b4
  %v50 = phi i8* [ %v45, %b4 ], [ %v76, %b5 ]
  %v51 = phi i32 [ %v44, %b4 ], [ %v75, %b5 ]
  %v52 = phi i32 [ %v26, %b4 ], [ %v53, %b5 ]
  %v53 = phi i32 [ %v32, %b4 ], [ %v64, %b5 ]
  %v54 = phi i8* [ %v18, %b4 ], [ %v50, %b5 ]
  %v55 = phi i8* [ %v47, %b4 ], [ %v61, %b5 ]
  %v56 = phi i8* [ %v48, %b4 ], [ %v57, %b5 ]
  %v57 = getelementptr inbounds i8, i8* %v56, i32 1
  %v58 = load i8, i8* %v56, align 1
  %v59 = zext i8 %v58 to i32
  %v60 = mul nsw i32 %v59, 3
  %v61 = getelementptr inbounds i8, i8* %v55, i32 1
  %v62 = load i8, i8* %v55, align 1
  %v63 = zext i8 %v62 to i32
  %v64 = add nsw i32 %v60, %v63
  %v65 = mul nsw i32 %v53, 3
  %v66 = add i32 %v52, 8
  %v67 = add i32 %v66, %v65
  %v68 = lshr i32 %v67, 4
  %v69 = trunc i32 %v68 to i8
  %v70 = getelementptr inbounds i8, i8* %v54, i32 3
  store i8 %v69, i8* %v50, align 1
  %v71 = add i32 %v65, 7
  %v72 = add i32 %v71, %v64
  %v73 = lshr i32 %v72, 4
  %v74 = trunc i32 %v73 to i8
  store i8 %v74, i8* %v70, align 1
  %v75 = add i32 %v51, -1
  %v76 = getelementptr inbounds i8, i8* %v50, i32 2
  %v77 = icmp eq i32 %v75, 0
  br i1 %v77, label %b6, label %b5

b6:                                               ; preds = %b5
  %v78 = add i32 %v49, -2
  %v79 = getelementptr i8, i8* %v18, i32 %v78
  %v80 = add i32 %v49, -4
  %v81 = getelementptr i8, i8* %v18, i32 %v80
  br label %b7

b7:                                               ; preds = %b6, %b3
  %v82 = phi i8* [ %v79, %b6 ], [ %v45, %b3 ]
  %v83 = phi i32 [ %v53, %b6 ], [ %v26, %b3 ]
  %v84 = phi i32 [ %v64, %b6 ], [ %v32, %b3 ]
  %v85 = phi i8* [ %v81, %b6 ], [ %v18, %b3 ]
  %v86 = mul nsw i32 %v84, 3
  %v87 = add i32 %v83, 8
  %v88 = add i32 %v87, %v86
  %v89 = lshr i32 %v88, 4
  %v90 = trunc i32 %v89 to i8
  %v91 = getelementptr inbounds i8, i8* %v85, i32 3
  store i8 %v90, i8* %v82, align 1
  %v92 = mul nsw i32 %v84, 4
  %v93 = add nsw i32 %v92, 7
  %v94 = lshr i32 %v93, 4
  %v95 = trunc i32 %v94 to i8
  store i8 %v95, i8* %v91, align 1
  %v96 = add nsw i32 %v8, 1
  %v97 = icmp eq i32 %v96, 2
  br i1 %v97, label %b8, label %b3

b8:                                               ; preds = %b7
  %v98 = add i32 %v5, 2
  %v99 = add nsw i32 %v6, 1
  %v100 = load i32, i32* %v1, align 4
  %v101 = icmp slt i32 %v98, %v100
  br i1 %v101, label %b2, label %b9

b9:                                               ; preds = %b8
  br label %b10

b10:                                              ; preds = %b9, %b0
  ret void
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv60" }
