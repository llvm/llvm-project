; RUN: llc -O3 -march=hexagon < %s | FileCheck %s

; CHECK-NOT: memb(r{{[0-9]+}}+#375) = #4
; CHECK: [[REG0:(r[0-9]+)]] = add(r{{[0-9]+}},{{#?}}#374)
; CHECK: memb([[REG0]]+#1) = #4

%s.0 = type { %s.1, %s.2*, %s.2*, %s.3, %s.5, i32, i32, i16, i8, i8, i8, [7 x i8], i16, i8, i8, i16, i8, i8, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i32, i16, i16, i16, [14 x i8], %s.6, i8, i8, %s.8, [2 x [16 x %s.9]], i32 (i8*, i8*, i8*, i8*, i8*)*, [80 x i8], i8, i8, i8*, i8*, i8*, i32*, i8*, i8*, i8*, [4 x i8], i8*, i8*, i8*, i8*, i8*, i8*, %s.18*, %s.18*, %s.6*, [4 x i8], [2 x [80 x [8 x i8]]], [56 x i8], [2 x [81 x %s.10]], [2 x %s.10], %s.10*, %s.10*, i32, [32 x i32], i8*, %s.12*, i8, i8, %s.18, i64*, i32, %s.19, %s.20, %s.21*, i8, [19 x i8] }
%s.1 = type { i32, i32, i8* }
%s.2 = type { i8, i8 }
%s.3 = type { [371 x %s.2], [6 x %s.4] }
%s.4 = type { %s.2*, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8 }
%s.5 = type { [12 x %s.2], [4 x %s.2], [2 x %s.2], [4 x %s.2], [6 x %s.2], [2 x [7 x %s.2]], [4 x %s.2], [3 x [4 x %s.2]], [3 x %s.2], [3 x %s.2] }
%s.6 = type { i8*, i32, %s.7, i8*, i8*, i32 }
%s.7 = type { i64 }
%s.8 = type { i8, i8, i8, i8, i8, i8, i8, i8, i32, i8, i8, [2 x i8], [16 x i8], [4 x i8], [32 x i16], [32 x i16], [4 x i8], [2 x [4 x i8]], [2 x [4 x i8]], i32, i32, i16, i8 }
%s.9 = type { [2 x i16] }
%s.10 = type { %s.11, [2 x [4 x %s.9]], [2 x [2 x i8]], [2 x i8] }
%s.11 = type { i8, i8, i8, i8, i8, i8, i8, i8, i32 }
%s.12 = type { i8*, i8*, i32, i8*, i16*, i8*, i16*, i8*, i32, i16, i8, i32, i16*, i16*, i16, i16, i16, i8, i8, %s.13, i8, i8, i8, [32 x i8*], %s.14, %s.16, i8, i8, i8, i8 }
%s.13 = type { [6 x [16 x i8]], [2 x [64 x i8]] }
%s.14 = type { i32, i32, %s.15* }
%s.15 = type { i16, i16 }
%s.16 = type { %s.17 }
%s.17 = type { i32, i32, i32 }
%s.18 = type { i16*, i16*, i16*, i16*, i16*, i32, i32 }
%s.19 = type { i32, i32, i32, i32 }
%s.20 = type { i32, i32, i32 }
%s.21 = type { %s.22*, i8, i8, i8*, i8*, i8*, i8*, i16, i8, void (%s.21*, i8, i8*, i8*, %s.25*, i32*)*, i8, i8, i8, i16, i8, i16, i8, i8, i8*, [4 x i8], i8, i8, [2 x i8], [2 x [4 x i8]], [2 x i16*], i8, i8, i8, i8, i16, i16, i16, i16, i16, i32, [4 x i8], [2 x %s.35], [2 x %s.35], [2 x [10 x %s.30]], %s.35*, %s.35*, %s.35*, %s.35*, [2 x %s.30*], [2 x %s.30*], [2 x %s.30*], [2 x %s.30*], %s.35, [2 x [16 x %s.30]], [2 x [5 x %s.30]], %s.37*, [4 x i8], %s.37, i8, i8, [6 x i8] }
%s.22 = type { void (%s.21*, %s.23*)*, %s.27*, %s.28, %s.32, [4 x i8], [2 x [81 x %s.34]], [52 x i8], [52 x i8] }
%s.23 = type { i16, i16, i8, [64 x %s.24], i8, i8, %s.26, [2 x i8], [4 x [2 x [4 x i16]]], [4 x [2 x [4 x i8]]], [32 x i8*], [32 x i8*] }
%s.24 = type { %s.25, i8, i8, i8, i8, i8, i8, i8, i16 }
%s.25 = type { i32 }
%s.26 = type { i8, i8, i8, [2 x [3 x [32 x i16]]], [2 x [3 x [32 x i8]]] }
%s.27 = type { i16, i16, i32, i8, [3 x i8], %s.13, i8, i8, [2 x i8], [1280 x i8], [765 x i8], [3 x i8], [2 x [640 x i8]], [2 x [720 x i8]], [80 x i8], [45 x i8], [45 x i8], [45 x i8] }
%s.28 = type { i8, i8, i8, i32, i32, i32, i32, i8, i8, i8, i8, i8, i8, i8, i8, i8, %s.13*, i8, i8, i16*, i8, i8, i8, i8*, %s.29*, %s.31* }
%s.29 = type { i8, %s.30*, i8* }
%s.30 = type { %s.25, i8, i8, i8, i8 }
%s.31 = type { i8, i8, i8, i8, %s.31**, i8, i8, i8, i8, i8, i8, i8, %s.31**, i8, %s.31**, i8*, i8*, i32, i32, i32, i32, i32, [2 x i8], [2 x i8], i32, i32, i8, i8, i32, %s.31*, %s.29*, i8, i8, i32, i32, i32, i32, i32, i32, [2 x i32], [2 x i64], [2 x i32], [2 x i32], [2 x i32], [2 x i32], [2 x i64] }
%s.32 = type { i8, i8, i16, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, i8, %s.26, %s.6*, [32 x %s.33], %s.33, [32 x %s.33], %s.29*, i8, [2 x [32 x i8]], [32 x i8*], [32 x i8*], [2 x [32 x i8]], [72 x i8], [72 x i32], [72 x i32], [72 x i32], [3 x [2 x [32 x [32 x i16]]]] }
%s.33 = type { i32, i32, i32, i8, i8 }
%s.34 = type { %s.35, [2 x [4 x %s.30]] }
%s.35 = type { i32, i16, %s.36, i8, [3 x i8], i32 }
%s.36 = type { i16 }
%s.37 = type { i8, [1 x %s.38], [1 x [416 x i16]], %s.40, %s.38*, %s.38*, i16*, [4 x i8], i16*, %s.40*, %s.40*, %s.40*, %s.27*, [4 x i8], %s.42, %s.23, %s.43, i8 }
%s.38 = type { %s.39, %s.39, %s.39 }
%s.39 = type { i8*, i16, i16, i16 }
%s.40 = type { %s.41, %s.41, %s.41, i8 }
%s.41 = type { i8*, i16, i16, i16 }
%s.42 = type { [32 x i8], [3 x i8], [3 x i8], [3 x i8], [3 x i8], [3 x i8], [3 x i8], i8, i8, [4 x i8] }
%s.43 = type { i32, i32, i32, i32, i32, [3 x i8], [3 x i8], [3 x i8], [16 x i8], i8, i8, i8, i8, i32, i32, i16, i16* }
%s.44 = type { i8, i8 }

; Function Attrs: nounwind
define i32 @f0(%s.0* %a0, %s.21* %a1, i8 zeroext %a2, %s.18* %a3, %s.20* %a4) local_unnamed_addr #0 {
b0:
  %v0 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 39, i32 2
  %v1 = load i8, i8* %v0, align 2
  %v2 = getelementptr inbounds %s.21, %s.21* %a1, i32 0, i32 47, i32 2
  %v3 = bitcast %s.36* %v2 to %s.44*
  %v4 = getelementptr inbounds %s.44, %s.44* %v3, i32 0, i32 1
  store i8 %v1, i8* %v4, align 1
  %v5 = getelementptr inbounds %s.21, %s.21* %a1, i32 0, i32 32
  %v6 = getelementptr inbounds %s.21, %s.21* %a1, i32 0, i32 16
  switch i8 %v1, label %b5 [
    i8 1, label %b1
    i8 0, label %b2
  ]

b1:                                               ; preds = %b0
  %v7 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 39, i32 10
  %v8 = load i8, i8* %v7, align 1
  %v9 = getelementptr inbounds %s.0, %s.0* %a0, i32 0, i32 39, i32 3
  %v10 = load i8, i8* %v9, align 1
  store i8 %v10, i8* %v6, align 2
  %v11 = getelementptr inbounds %s.21, %s.21* %a1, i32 0, i32 19
  %v12 = bitcast [4 x i8]* %v11 to i32*
  store i32 16843009, i32* %v12, align 8
  %v13 = icmp eq i8 %v10, 15
  switch i8 %v1, label %b4 [
    i8 6, label %b3
    i8 1, label %b3
  ]

b2:                                               ; preds = %b0
  store i8 4, i8* %v4, align 1
  store i8 0, i8* %v6, align 2
  switch i8 %v1, label %b4 [
    i8 6, label %b3
    i8 1, label %b3
  ]

b3:                                               ; preds = %b2, %b2, %b1, %b1
  %v14 = tail call fastcc signext i8 @f1(%s.21* nonnull %a1)
  unreachable

b4:                                               ; preds = %b2, %b1
  unreachable

b5:                                               ; preds = %b0
  unreachable
}

; Function Attrs: norecurse nounwind
declare i8 @f1(%s.21* nocapture) unnamed_addr #1

attributes #0 = { nounwind "target-cpu"="hexagonv65" }
attributes #1 = { norecurse nounwind "target-cpu"="hexagonv65" }
