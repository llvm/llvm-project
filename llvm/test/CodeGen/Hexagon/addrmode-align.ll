; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK: [[REG0:(r[0-9]+)]] = add(r29
; CHECK: [[REG1:(r[0-9]+)]] = add([[REG0]],#8)
; CHECK-DAG: memd([[REG1]]+#8) =
; CHECK-DAG: memd([[REG1]]+#0) =

%s.0 = type { i32, i8, double, i32, float }

@g0 = external local_unnamed_addr global i32, align 4

define i32 @f0() local_unnamed_addr {
b0:
  %v0 = alloca [10 x %s.0], align 8
  %v1 = bitcast [10 x %s.0]* %v0 to i8*
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v2 = phi i32 [ 0, %b0 ], [ %v6, %b1 ]
  %v3 = getelementptr inbounds [10 x %s.0], [10 x %s.0]* %v0, i32 0, i32 %v2, i32 0
  store i32 0, i32* %v3, align 8
  %v4 = getelementptr inbounds [10 x %s.0], [10 x %s.0]* %v0, i32 0, i32 %v2, i32 1
  store i8 0, i8* %v4, align 4
  %v5 = getelementptr inbounds [10 x %s.0], [10 x %s.0]* %v0, i32 0, i32 %v2, i32 2
  %v6 = add nuw nsw i32 %v2, 1
  %v7 = icmp eq i32 %v6, 10
  %v8 = bitcast double* %v5 to i8*
  call void @llvm.memset.p0i8.i64(i8* align 8 %v8, i8 0, i64 16, i1 false)
  br i1 %v7, label %b2, label %b1

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v9 = phi i32 [ 0, %b2 ], [ %v10, %b3 ]
  %v10 = add nuw nsw i32 %v9, 1
  %v11 = icmp eq i32 %v10, 10
  br i1 %v11, label %b4, label %b3

b4:                                               ; preds = %b3
  %v12 = getelementptr inbounds [10 x %s.0], [10 x %s.0]* %v0, i32 0, i32 0, i32 0
  %v13 = load i32, i32* %v12, align 8
  %v14 = sub nsw i32 1122, %v13
  %v15 = icmp eq i32 %v14, 1121
  br i1 %v15, label %b6, label %b5

b5:                                               ; preds = %b4
  store i32 1, i32* @g0, align 4
  br label %b6

b6:                                               ; preds = %b5, %b4
  tail call void @f1()
  unreachable
}

declare void @f1() local_unnamed_addr

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #0

attributes #0 = { argmemonly nounwind }
