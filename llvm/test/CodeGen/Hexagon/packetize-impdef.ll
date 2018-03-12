; RUN: llc -O3 -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts
;
; Check that IMPLICIT_DEFs are packetized correctly
; (previously caused an assert).
;
; CHECK: f1:

%0 = type { i8 (i8)*, i8 (i8, %1*)*, i8 (i8)* }
%1 = type { [16384 x i16], [8192 x i16], [8192 x i16], [8192 x i32], i32, i32, i32, %2, %2, i32, i32, i32, i32 }
%2 = type { i32, i32, i32 }
%3 = type { %4 }
%4 = type { i32, i8* }
%5 = type { i8, i32, i32, i32, i16, i16, i16, i16, i8, i16, %6, %6, i32, i16, i16, i16, i16, i8 }
%6 = type { i32, i32, i32, i32, i32, i32, i32, i8, i8 }
%7 = type { i8, i8, i8, i8, i32, i32, i32, i32, i32, i32, %2, %2, %8, i8 }
%8 = type { %2, %2 }

@g0 = external hidden unnamed_addr constant [7 x %0], align 8
@g1 = external hidden global %1, align 4
@g2 = external hidden constant %3, align 4
@g3 = external hidden constant %3, align 4

declare void @f0(%3*, i32, i32)

define hidden fastcc i32 @f1(%5* %a0, %7* %a1, %2* %a2) {
b0:
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b0
  unreachable

b2:                                               ; preds = %b0
  br i1 undef, label %b3, label %b4

b3:                                               ; preds = %b2
  br label %b55

b4:                                               ; preds = %b2
  br i1 undef, label %b6, label %b5

b5:                                               ; preds = %b4
  %v0 = getelementptr inbounds %5, %5* %a0, i32 0, i32 1
  br label %b7

b6:                                               ; preds = %b4
  br label %b55

b7:                                               ; preds = %b52, %b5
  %v1 = phi i32 [ undef, %b5 ], [ %v43, %b52 ]
  %v2 = phi i32 [ 5, %b5 ], [ %v45, %b52 ]
  %v3 = load i32, i32* undef, align 4
  %v4 = load i32, i32* %v0, align 4
  %v5 = sext i32 %v4 to i64
  %v6 = sdiv i64 0, %v5
  %v7 = trunc i64 %v6 to i32
  %v8 = icmp slt i32 %v7, 204800
  br i1 %v8, label %b8, label %b9

b8:                                               ; preds = %b7
  call void @f0(%3* @g2, i32 %v3, i32 %v4)
  br label %b54

b9:                                               ; preds = %b7
  %v9 = load i8, i8* undef, align 1
  %v10 = zext i8 %v9 to i32
  br i1 undef, label %b10, label %b11

b10:                                              ; preds = %b9
  br label %b47

b11:                                              ; preds = %b9
  br i1 undef, label %b12, label %b47

b12:                                              ; preds = %b11
  br i1 undef, label %b13, label %b47

b13:                                              ; preds = %b12
  %v11 = getelementptr inbounds [7 x %0], [7 x %0]* @g0, i32 0, i32 %v10, i32 2
  %v12 = load i8 (i8)*, i8 (i8)** %v11, align 4
  %v13 = call zeroext i8 %v12(i8 zeroext %v9)
  br i1 undef, label %b14, label %b47

b14:                                              ; preds = %b13
  br i1 undef, label %b15, label %b16

b15:                                              ; preds = %b14
  br label %b46

b16:                                              ; preds = %b14
  br i1 false, label %b17, label %b22

b17:                                              ; preds = %b16
  br i1 undef, label %b18, label %b19

b18:                                              ; preds = %b17
  unreachable

b19:                                              ; preds = %b17
  br label %b20

b20:                                              ; preds = %b20, %b19
  br i1 undef, label %b20, label %b21

b21:                                              ; preds = %b20
  unreachable

b22:                                              ; preds = %b16
  br i1 false, label %b23, label %b24

b23:                                              ; preds = %b22
  br label %b47

b24:                                              ; preds = %b22
  br i1 false, label %b25, label %b26

b25:                                              ; preds = %b24
  unreachable

b26:                                              ; preds = %b24
  br label %b27

b27:                                              ; preds = %b36, %b26
  %v14 = phi i32 [ 16, %b26 ], [ %v30, %b36 ]
  %v15 = getelementptr inbounds %1, %1* @g1, i32 0, i32 2, i32 %v14
  %v16 = load i16, i16* %v15, align 2
  %v17 = sext i16 %v16 to i32
  %v18 = select i1 undef, i32 undef, i32 %v17
  %v19 = sext i32 %v18 to i64
  %v20 = or i32 %v18, undef
  br i1 false, label %b28, label %b29

b28:                                              ; preds = %b27
  unreachable

b29:                                              ; preds = %b27
  br i1 false, label %b30, label %b31

b30:                                              ; preds = %b29
  unreachable

b31:                                              ; preds = %b29
  %v21 = mul nsw i64 undef, %v19
  %v22 = sdiv i64 0, %v19
  %v23 = add nsw i64 %v22, 0
  %v24 = lshr i64 %v23, 5
  %v25 = trunc i64 %v24 to i32
  %v26 = sub nsw i32 1608, %v25
  %v27 = icmp sgt i16 %v16, -1
  %v28 = and i1 undef, %v27
  br i1 %v28, label %b32, label %b33

b32:                                              ; preds = %b31
  store i32 %v26, i32* undef, align 4
  br label %b36

b33:                                              ; preds = %b31
  br i1 undef, label %b34, label %b35

b34:                                              ; preds = %b33
  %v29 = getelementptr inbounds %1, %1* @g1, i32 0, i32 3, i32 %v14
  store i32 undef, i32* %v29, align 4
  br label %b36

b35:                                              ; preds = %b33
  br label %b36

b36:                                              ; preds = %b35, %b34, %b32
  %v30 = add nuw nsw i32 %v14, 1
  %v31 = icmp ult i32 %v30, 8192
  br i1 %v31, label %b27, label %b37

b37:                                              ; preds = %b36
  br label %b38

b38:                                              ; preds = %b38, %b37
  br i1 undef, label %b38, label %b39

b39:                                              ; preds = %b38
  br i1 false, label %b40, label %b41

b40:                                              ; preds = %b39
  unreachable

b41:                                              ; preds = %b39
  %v32 = icmp ult i8 %v9, 6
  br i1 %v32, label %b43, label %b42

b42:                                              ; preds = %b41
  br label %b47

b43:                                              ; preds = %b41
  %v33 = load i64, i64* undef, align 8
  br label %b44

b44:                                              ; preds = %b44, %b43
  br i1 undef, label %b45, label %b44

b45:                                              ; preds = %b44
  %v34 = sdiv i64 undef, %v33
  %v35 = trunc i64 %v34 to i32
  %v36 = add nsw i32 0, %v3
  %v37 = sext i32 %v36 to i64
  %v38 = mul nsw i64 %v37, 4096000
  %v39 = sdiv i64 %v38, 0
  %v40 = trunc i64 %v39 to i32
  br label %b46

b46:                                              ; preds = %b45, %b15
  %v41 = phi i32 [ undef, %b15 ], [ %v40, %b45 ]
  br label %b47

b47:                                              ; preds = %b46, %b42, %b23, %b13, %b12, %b11, %b10
  %v42 = phi i8 [ 1, %b10 ], [ 0, %b46 ], [ 3, %b23 ], [ 1, %b42 ], [ %v13, %b13 ], [ undef, %b12 ], [ undef, %b11 ]
  %v43 = phi i32 [ %v1, %b10 ], [ %v41, %b46 ], [ %v1, %b23 ], [ %v1, %b42 ], [ %v1, %b13 ], [ %v1, %b12 ], [ %v1, %b11 ]
  %v44 = icmp eq i8 %v42, 1
  br i1 %v44, label %b48, label %b49

b48:                                              ; preds = %b47
  br label %b54

b49:                                              ; preds = %b47
  br i1 undef, label %b50, label %b52

b50:                                              ; preds = %b49
  br i1 undef, label %b51, label %b53

b51:                                              ; preds = %b50
  br label %b52

b52:                                              ; preds = %b51, %b49
  %v45 = add nsw i32 %v2, -1
  %v46 = icmp eq i32 %v45, 0
  br i1 %v46, label %b54, label %b7

b53:                                              ; preds = %b50
  call void @f0(%3* @g3, i32 %v43, i32 undef)
  unreachable

b54:                                              ; preds = %b52, %b48, %b8
  unreachable

b55:                                              ; preds = %b6, %b3
  ret i32 0
}
