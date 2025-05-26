; RUN: llc -march=hexagon -mv71 -O2 < %s -o - 2>&1 > /dev/null
; Ensure we do not invalidate a DAG by forming a circuit.
; If we form a circuit, this test crashes while creating the DAG
; with topological sorting.

%struct.quux = type { i16, i16, i8, i8, i8, i8, i8, i8, i8, i8, [2 x i8], %struct.ham, %struct.bar, i8 }
%struct.ham = type { i8, [2 x i8], [2 x i8], [2 x i8] }
%struct.bar = type { [2 x i8], [2 x i8], [2 x i8] }

define dso_local void @blam(i32 %arg, i8 %dummy, i32 %tmp) local_unnamed_addr #0 {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %phi = phi i8 [ %phi6, %bb1 ], [ %dummy, %bb ]
  %phi2 = phi i8 [ %phi, %bb1 ], [ %dummy, %bb ]
  %phi3 = phi i8 [ %phi2, %bb1 ], [ 0, %bb ]
  %phi4 = phi i8 [ %phi3, %bb1 ], [ %dummy, %bb ]
  %phi5 = phi i8 [ %phi4, %bb1 ], [ %dummy, %bb ]
  %phi6 = phi i8 [ %phi5, %bb1 ], [ %dummy, %bb ]
  %phi7 = phi i32 [ %add, %bb1 ], [ %tmp, %bb ]
  %getelementptr = getelementptr inbounds %struct.quux, ptr null, i32 %arg, i32 12, i32 1, i8 %dummy
  store i8 %phi4, ptr %getelementptr, align 1
  %add = add i32 %phi7, -1
  %icmp = icmp eq i32 %add, 0
  br i1 %icmp, label %bb8, label %bb1

bb8:                                              ; preds = %bb1
  ret void
}

attributes #0 = { "target-features"="+v71,-long-calls,-small-data" }
