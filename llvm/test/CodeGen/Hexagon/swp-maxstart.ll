; RUN: llc -march=hexagon -O3 < %s -pipeliner-experimental-cg=true | FileCheck %s

; Test that the MinStart computation, which is based upon the length
; of the chain edges, is computed correctly. A bug in the code allowed
; two instuctions that have a chain edge to be scheduled more than II
; instructions apart. In this test, if two stores appear before the
; store, then that is a bug.

; CHECK: r{{[0-9]+}} = memw([[REG0:r([0-9]+)]]+#12)
; CHECK-NOT: r{{[0-9]+}} = memw([[REG0]]+#12)
; CHECK: memw([[REG0]]+#12) = r{{[0-9]+}}

%s.0 = type { i64, i32, i32, i32, ptr }

@g0 = external global %s.0, align 8

; Function Attrs: nounwind
define void @f0() #0 {
b0:
  %v0 = load i32, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 1), align 8
  %v1 = ashr i32 %v0, 3
  br i1 undef, label %b1, label %b2

b1:                                               ; preds = %b1, %b0
  %v2 = phi i32 [ %v5, %b1 ], [ 0, %b0 ]
  %v3 = load ptr, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 4), align 4
  %v4 = getelementptr inbounds i8, ptr %v3, i32 -1
  store ptr %v4, ptr getelementptr inbounds (%s.0, ptr @g0, i32 0, i32 4), align 4
  store i8 0, ptr %v4, align 1
  %v5 = add nsw i32 %v2, 1
  %v6 = icmp eq i32 %v5, %v1
  br i1 %v6, label %b2, label %b1

b2:                                               ; preds = %b1, %b0
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
