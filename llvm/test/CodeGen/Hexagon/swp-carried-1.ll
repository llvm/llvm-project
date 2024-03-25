; RUN: llc -march=hexagon -rdf-opt=0 -disable-hexagon-misched -hexagon-initial-cfg-cleanup=0 -lsr-setupcost-depth-limit=1 -disable-cgp-delete-phis < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that we generate the correct code when a loop carried value
; is scheduled one stage earlier than it's use. The code in
; isLoopCarried was returning false in this case, and the generated
; code was missing an copy.

; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: += mpy([[REG0:(r[0-9]+)]],r{{[0-9]+}})
; CHECK: [[REG0]] = r{{[0-9]+}}
; CHECK-NOT: [[REG0]] = memw
; CHECK: endloop0

@g0 = external global [256 x i32], align 8

define void @f0() #0 {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  br i1 undef, label %b2, label %b1

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b3, %b2
  %v0 = phi ptr [ @g0, %b2 ], [ %v1, %b3 ]
  %v1 = getelementptr i32, ptr %v0, i32 6
  br i1 undef, label %b4, label %b3

b4:                                               ; preds = %b3
  br i1 undef, label %b6, label %b5

b5:                                               ; preds = %b5, %b4
  %v2 = phi i64 [ %v19, %b5 ], [ undef, %b4 ]
  %v3 = phi ptr [ %v8, %b5 ], [ %v1, %b4 ]
  %v4 = phi i32 [ %v9, %b5 ], [ undef, %b4 ]
  %v5 = phi i32 [ %v11, %b5 ], [ undef, %b4 ]
  %v6 = phi i32 [ %v5, %b5 ], [ undef, %b4 ]
  %v7 = phi i32 [ %v10, %b5 ], [ 0, %b4 ]
  %v8 = getelementptr i32, ptr %v3, i32 1
  %v9 = add nsw i32 %v4, 1
  %v10 = load i32, ptr %v8, align 4
  %v11 = load i32, ptr null, align 4
  %v12 = sext i32 %v6 to i64
  %v13 = sext i32 %v10 to i64
  %v14 = sext i32 %v7 to i64
  %v15 = mul nsw i64 %v14, %v12
  %v16 = add i64 %v12, %v2
  %v17 = add i64 %v16, %v13
  %v18 = add i64 %v17, 0
  %v19 = add i64 %v18, %v15
  %v20 = icmp eq i32 %v9, 128
  br i1 %v20, label %b6, label %b5

b6:                                               ; preds = %b5, %b4
  %v21 = phi i64 [ undef, %b4 ], [ %v19, %b5 ]
  unreachable
}

attributes #0 = { nounwind "target-cpu"="hexagonv62" }
