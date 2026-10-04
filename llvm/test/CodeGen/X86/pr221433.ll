; RUN: llc -mtriple=x86_64-- < %s | FileCheck %s

; https://github.com/llvm/llvm-project/issues/221433
; Compile-time test: a block of 320 independent strided stores (row stride of
; 20004 bytes, i.e. arr[i][i] with a non-power-of-two dimension). Store chain
; improvement repeatedly rebuilds the wide TokenFactor joining the stores, and
; DAGCombiner used to revisit all of its operands on every rebuild. That made
; this test take seconds (and the full 1000-store repro from the issue hang).

; CHECK-LABEL: test:
; CHECK: retq

@arr = external global [5000 x [5000 x i32]], align 16

define void @test() {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %g0.1 = or disjoint i64 %i, 1
  %g0.2 = or disjoint i64 %i, 2
  %g0.3 = or disjoint i64 %i, 3
  %g0.4 = or disjoint i64 %i, 4
  %g0.5 = or disjoint i64 %i, 5
  %g0.6 = or disjoint i64 %i, 6
  %g0.7 = or disjoint i64 %i, 7
  %row0.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i
  %p0.0 = getelementptr inbounds nuw [4 x i8], ptr %row0.0, i64 %i
  store i32 0, ptr %p0.0, align 4
  %row0.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g0.1
  %p0.1 = getelementptr inbounds nuw [4 x i8], ptr %row0.1, i64 %g0.1
  store i32 0, ptr %p0.1, align 4
  %row0.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g0.2
  %p0.2 = getelementptr inbounds nuw [4 x i8], ptr %row0.2, i64 %g0.2
  store i32 0, ptr %p0.2, align 4
  %row0.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g0.3
  %p0.3 = getelementptr inbounds nuw [4 x i8], ptr %row0.3, i64 %g0.3
  store i32 0, ptr %p0.3, align 4
  %row0.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g0.4
  %p0.4 = getelementptr inbounds nuw [4 x i8], ptr %row0.4, i64 %g0.4
  store i32 0, ptr %p0.4, align 4
  %row0.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g0.5
  %p0.5 = getelementptr inbounds nuw [4 x i8], ptr %row0.5, i64 %g0.5
  store i32 0, ptr %p0.5, align 4
  %row0.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g0.6
  %p0.6 = getelementptr inbounds nuw [4 x i8], ptr %row0.6, i64 %g0.6
  store i32 0, ptr %p0.6, align 4
  %row0.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g0.7
  %p0.7 = getelementptr inbounds nuw [4 x i8], ptr %row0.7, i64 %g0.7
  store i32 0, ptr %p0.7, align 4
  %i.1 = add nuw nsw i64 %i, 8
  %g1.1 = or disjoint i64 %i.1, 1
  %g1.2 = or disjoint i64 %i.1, 2
  %g1.3 = or disjoint i64 %i.1, 3
  %g1.4 = or disjoint i64 %i.1, 4
  %g1.5 = or disjoint i64 %i.1, 5
  %g1.6 = or disjoint i64 %i.1, 6
  %g1.7 = or disjoint i64 %i.1, 7
  %row1.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.1
  %p1.0 = getelementptr inbounds nuw [4 x i8], ptr %row1.0, i64 %i.1
  store i32 0, ptr %p1.0, align 4
  %row1.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g1.1
  %p1.1 = getelementptr inbounds nuw [4 x i8], ptr %row1.1, i64 %g1.1
  store i32 0, ptr %p1.1, align 4
  %row1.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g1.2
  %p1.2 = getelementptr inbounds nuw [4 x i8], ptr %row1.2, i64 %g1.2
  store i32 0, ptr %p1.2, align 4
  %row1.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g1.3
  %p1.3 = getelementptr inbounds nuw [4 x i8], ptr %row1.3, i64 %g1.3
  store i32 0, ptr %p1.3, align 4
  %row1.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g1.4
  %p1.4 = getelementptr inbounds nuw [4 x i8], ptr %row1.4, i64 %g1.4
  store i32 0, ptr %p1.4, align 4
  %row1.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g1.5
  %p1.5 = getelementptr inbounds nuw [4 x i8], ptr %row1.5, i64 %g1.5
  store i32 0, ptr %p1.5, align 4
  %row1.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g1.6
  %p1.6 = getelementptr inbounds nuw [4 x i8], ptr %row1.6, i64 %g1.6
  store i32 0, ptr %p1.6, align 4
  %row1.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g1.7
  %p1.7 = getelementptr inbounds nuw [4 x i8], ptr %row1.7, i64 %g1.7
  store i32 0, ptr %p1.7, align 4
  %i.2 = add nuw nsw i64 %i, 16
  %g2.1 = or disjoint i64 %i.2, 1
  %g2.2 = or disjoint i64 %i.2, 2
  %g2.3 = or disjoint i64 %i.2, 3
  %g2.4 = or disjoint i64 %i.2, 4
  %g2.5 = or disjoint i64 %i.2, 5
  %g2.6 = or disjoint i64 %i.2, 6
  %g2.7 = or disjoint i64 %i.2, 7
  %row2.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.2
  %p2.0 = getelementptr inbounds nuw [4 x i8], ptr %row2.0, i64 %i.2
  store i32 0, ptr %p2.0, align 4
  %row2.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g2.1
  %p2.1 = getelementptr inbounds nuw [4 x i8], ptr %row2.1, i64 %g2.1
  store i32 0, ptr %p2.1, align 4
  %row2.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g2.2
  %p2.2 = getelementptr inbounds nuw [4 x i8], ptr %row2.2, i64 %g2.2
  store i32 0, ptr %p2.2, align 4
  %row2.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g2.3
  %p2.3 = getelementptr inbounds nuw [4 x i8], ptr %row2.3, i64 %g2.3
  store i32 0, ptr %p2.3, align 4
  %row2.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g2.4
  %p2.4 = getelementptr inbounds nuw [4 x i8], ptr %row2.4, i64 %g2.4
  store i32 0, ptr %p2.4, align 4
  %row2.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g2.5
  %p2.5 = getelementptr inbounds nuw [4 x i8], ptr %row2.5, i64 %g2.5
  store i32 0, ptr %p2.5, align 4
  %row2.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g2.6
  %p2.6 = getelementptr inbounds nuw [4 x i8], ptr %row2.6, i64 %g2.6
  store i32 0, ptr %p2.6, align 4
  %row2.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g2.7
  %p2.7 = getelementptr inbounds nuw [4 x i8], ptr %row2.7, i64 %g2.7
  store i32 0, ptr %p2.7, align 4
  %i.3 = add nuw nsw i64 %i, 24
  %g3.1 = or disjoint i64 %i.3, 1
  %g3.2 = or disjoint i64 %i.3, 2
  %g3.3 = or disjoint i64 %i.3, 3
  %g3.4 = or disjoint i64 %i.3, 4
  %g3.5 = or disjoint i64 %i.3, 5
  %g3.6 = or disjoint i64 %i.3, 6
  %g3.7 = or disjoint i64 %i.3, 7
  %row3.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.3
  %p3.0 = getelementptr inbounds nuw [4 x i8], ptr %row3.0, i64 %i.3
  store i32 0, ptr %p3.0, align 4
  %row3.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g3.1
  %p3.1 = getelementptr inbounds nuw [4 x i8], ptr %row3.1, i64 %g3.1
  store i32 0, ptr %p3.1, align 4
  %row3.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g3.2
  %p3.2 = getelementptr inbounds nuw [4 x i8], ptr %row3.2, i64 %g3.2
  store i32 0, ptr %p3.2, align 4
  %row3.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g3.3
  %p3.3 = getelementptr inbounds nuw [4 x i8], ptr %row3.3, i64 %g3.3
  store i32 0, ptr %p3.3, align 4
  %row3.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g3.4
  %p3.4 = getelementptr inbounds nuw [4 x i8], ptr %row3.4, i64 %g3.4
  store i32 0, ptr %p3.4, align 4
  %row3.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g3.5
  %p3.5 = getelementptr inbounds nuw [4 x i8], ptr %row3.5, i64 %g3.5
  store i32 0, ptr %p3.5, align 4
  %row3.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g3.6
  %p3.6 = getelementptr inbounds nuw [4 x i8], ptr %row3.6, i64 %g3.6
  store i32 0, ptr %p3.6, align 4
  %row3.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g3.7
  %p3.7 = getelementptr inbounds nuw [4 x i8], ptr %row3.7, i64 %g3.7
  store i32 0, ptr %p3.7, align 4
  %i.4 = add nuw nsw i64 %i, 32
  %g4.1 = or disjoint i64 %i.4, 1
  %g4.2 = or disjoint i64 %i.4, 2
  %g4.3 = or disjoint i64 %i.4, 3
  %g4.4 = or disjoint i64 %i.4, 4
  %g4.5 = or disjoint i64 %i.4, 5
  %g4.6 = or disjoint i64 %i.4, 6
  %g4.7 = or disjoint i64 %i.4, 7
  %row4.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.4
  %p4.0 = getelementptr inbounds nuw [4 x i8], ptr %row4.0, i64 %i.4
  store i32 0, ptr %p4.0, align 4
  %row4.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g4.1
  %p4.1 = getelementptr inbounds nuw [4 x i8], ptr %row4.1, i64 %g4.1
  store i32 0, ptr %p4.1, align 4
  %row4.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g4.2
  %p4.2 = getelementptr inbounds nuw [4 x i8], ptr %row4.2, i64 %g4.2
  store i32 0, ptr %p4.2, align 4
  %row4.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g4.3
  %p4.3 = getelementptr inbounds nuw [4 x i8], ptr %row4.3, i64 %g4.3
  store i32 0, ptr %p4.3, align 4
  %row4.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g4.4
  %p4.4 = getelementptr inbounds nuw [4 x i8], ptr %row4.4, i64 %g4.4
  store i32 0, ptr %p4.4, align 4
  %row4.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g4.5
  %p4.5 = getelementptr inbounds nuw [4 x i8], ptr %row4.5, i64 %g4.5
  store i32 0, ptr %p4.5, align 4
  %row4.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g4.6
  %p4.6 = getelementptr inbounds nuw [4 x i8], ptr %row4.6, i64 %g4.6
  store i32 0, ptr %p4.6, align 4
  %row4.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g4.7
  %p4.7 = getelementptr inbounds nuw [4 x i8], ptr %row4.7, i64 %g4.7
  store i32 0, ptr %p4.7, align 4
  %i.5 = add nuw nsw i64 %i, 40
  %g5.1 = or disjoint i64 %i.5, 1
  %g5.2 = or disjoint i64 %i.5, 2
  %g5.3 = or disjoint i64 %i.5, 3
  %g5.4 = or disjoint i64 %i.5, 4
  %g5.5 = or disjoint i64 %i.5, 5
  %g5.6 = or disjoint i64 %i.5, 6
  %g5.7 = or disjoint i64 %i.5, 7
  %row5.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.5
  %p5.0 = getelementptr inbounds nuw [4 x i8], ptr %row5.0, i64 %i.5
  store i32 0, ptr %p5.0, align 4
  %row5.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g5.1
  %p5.1 = getelementptr inbounds nuw [4 x i8], ptr %row5.1, i64 %g5.1
  store i32 0, ptr %p5.1, align 4
  %row5.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g5.2
  %p5.2 = getelementptr inbounds nuw [4 x i8], ptr %row5.2, i64 %g5.2
  store i32 0, ptr %p5.2, align 4
  %row5.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g5.3
  %p5.3 = getelementptr inbounds nuw [4 x i8], ptr %row5.3, i64 %g5.3
  store i32 0, ptr %p5.3, align 4
  %row5.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g5.4
  %p5.4 = getelementptr inbounds nuw [4 x i8], ptr %row5.4, i64 %g5.4
  store i32 0, ptr %p5.4, align 4
  %row5.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g5.5
  %p5.5 = getelementptr inbounds nuw [4 x i8], ptr %row5.5, i64 %g5.5
  store i32 0, ptr %p5.5, align 4
  %row5.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g5.6
  %p5.6 = getelementptr inbounds nuw [4 x i8], ptr %row5.6, i64 %g5.6
  store i32 0, ptr %p5.6, align 4
  %row5.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g5.7
  %p5.7 = getelementptr inbounds nuw [4 x i8], ptr %row5.7, i64 %g5.7
  store i32 0, ptr %p5.7, align 4
  %i.6 = add nuw nsw i64 %i, 48
  %g6.1 = or disjoint i64 %i.6, 1
  %g6.2 = or disjoint i64 %i.6, 2
  %g6.3 = or disjoint i64 %i.6, 3
  %g6.4 = or disjoint i64 %i.6, 4
  %g6.5 = or disjoint i64 %i.6, 5
  %g6.6 = or disjoint i64 %i.6, 6
  %g6.7 = or disjoint i64 %i.6, 7
  %row6.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.6
  %p6.0 = getelementptr inbounds nuw [4 x i8], ptr %row6.0, i64 %i.6
  store i32 0, ptr %p6.0, align 4
  %row6.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g6.1
  %p6.1 = getelementptr inbounds nuw [4 x i8], ptr %row6.1, i64 %g6.1
  store i32 0, ptr %p6.1, align 4
  %row6.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g6.2
  %p6.2 = getelementptr inbounds nuw [4 x i8], ptr %row6.2, i64 %g6.2
  store i32 0, ptr %p6.2, align 4
  %row6.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g6.3
  %p6.3 = getelementptr inbounds nuw [4 x i8], ptr %row6.3, i64 %g6.3
  store i32 0, ptr %p6.3, align 4
  %row6.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g6.4
  %p6.4 = getelementptr inbounds nuw [4 x i8], ptr %row6.4, i64 %g6.4
  store i32 0, ptr %p6.4, align 4
  %row6.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g6.5
  %p6.5 = getelementptr inbounds nuw [4 x i8], ptr %row6.5, i64 %g6.5
  store i32 0, ptr %p6.5, align 4
  %row6.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g6.6
  %p6.6 = getelementptr inbounds nuw [4 x i8], ptr %row6.6, i64 %g6.6
  store i32 0, ptr %p6.6, align 4
  %row6.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g6.7
  %p6.7 = getelementptr inbounds nuw [4 x i8], ptr %row6.7, i64 %g6.7
  store i32 0, ptr %p6.7, align 4
  %i.7 = add nuw nsw i64 %i, 56
  %g7.1 = or disjoint i64 %i.7, 1
  %g7.2 = or disjoint i64 %i.7, 2
  %g7.3 = or disjoint i64 %i.7, 3
  %g7.4 = or disjoint i64 %i.7, 4
  %g7.5 = or disjoint i64 %i.7, 5
  %g7.6 = or disjoint i64 %i.7, 6
  %g7.7 = or disjoint i64 %i.7, 7
  %row7.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.7
  %p7.0 = getelementptr inbounds nuw [4 x i8], ptr %row7.0, i64 %i.7
  store i32 0, ptr %p7.0, align 4
  %row7.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g7.1
  %p7.1 = getelementptr inbounds nuw [4 x i8], ptr %row7.1, i64 %g7.1
  store i32 0, ptr %p7.1, align 4
  %row7.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g7.2
  %p7.2 = getelementptr inbounds nuw [4 x i8], ptr %row7.2, i64 %g7.2
  store i32 0, ptr %p7.2, align 4
  %row7.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g7.3
  %p7.3 = getelementptr inbounds nuw [4 x i8], ptr %row7.3, i64 %g7.3
  store i32 0, ptr %p7.3, align 4
  %row7.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g7.4
  %p7.4 = getelementptr inbounds nuw [4 x i8], ptr %row7.4, i64 %g7.4
  store i32 0, ptr %p7.4, align 4
  %row7.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g7.5
  %p7.5 = getelementptr inbounds nuw [4 x i8], ptr %row7.5, i64 %g7.5
  store i32 0, ptr %p7.5, align 4
  %row7.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g7.6
  %p7.6 = getelementptr inbounds nuw [4 x i8], ptr %row7.6, i64 %g7.6
  store i32 0, ptr %p7.6, align 4
  %row7.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g7.7
  %p7.7 = getelementptr inbounds nuw [4 x i8], ptr %row7.7, i64 %g7.7
  store i32 0, ptr %p7.7, align 4
  %i.8 = add nuw nsw i64 %i, 64
  %g8.1 = or disjoint i64 %i.8, 1
  %g8.2 = or disjoint i64 %i.8, 2
  %g8.3 = or disjoint i64 %i.8, 3
  %g8.4 = or disjoint i64 %i.8, 4
  %g8.5 = or disjoint i64 %i.8, 5
  %g8.6 = or disjoint i64 %i.8, 6
  %g8.7 = or disjoint i64 %i.8, 7
  %row8.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.8
  %p8.0 = getelementptr inbounds nuw [4 x i8], ptr %row8.0, i64 %i.8
  store i32 0, ptr %p8.0, align 4
  %row8.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g8.1
  %p8.1 = getelementptr inbounds nuw [4 x i8], ptr %row8.1, i64 %g8.1
  store i32 0, ptr %p8.1, align 4
  %row8.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g8.2
  %p8.2 = getelementptr inbounds nuw [4 x i8], ptr %row8.2, i64 %g8.2
  store i32 0, ptr %p8.2, align 4
  %row8.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g8.3
  %p8.3 = getelementptr inbounds nuw [4 x i8], ptr %row8.3, i64 %g8.3
  store i32 0, ptr %p8.3, align 4
  %row8.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g8.4
  %p8.4 = getelementptr inbounds nuw [4 x i8], ptr %row8.4, i64 %g8.4
  store i32 0, ptr %p8.4, align 4
  %row8.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g8.5
  %p8.5 = getelementptr inbounds nuw [4 x i8], ptr %row8.5, i64 %g8.5
  store i32 0, ptr %p8.5, align 4
  %row8.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g8.6
  %p8.6 = getelementptr inbounds nuw [4 x i8], ptr %row8.6, i64 %g8.6
  store i32 0, ptr %p8.6, align 4
  %row8.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g8.7
  %p8.7 = getelementptr inbounds nuw [4 x i8], ptr %row8.7, i64 %g8.7
  store i32 0, ptr %p8.7, align 4
  %i.9 = add nuw nsw i64 %i, 72
  %g9.1 = or disjoint i64 %i.9, 1
  %g9.2 = or disjoint i64 %i.9, 2
  %g9.3 = or disjoint i64 %i.9, 3
  %g9.4 = or disjoint i64 %i.9, 4
  %g9.5 = or disjoint i64 %i.9, 5
  %g9.6 = or disjoint i64 %i.9, 6
  %g9.7 = or disjoint i64 %i.9, 7
  %row9.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.9
  %p9.0 = getelementptr inbounds nuw [4 x i8], ptr %row9.0, i64 %i.9
  store i32 0, ptr %p9.0, align 4
  %row9.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g9.1
  %p9.1 = getelementptr inbounds nuw [4 x i8], ptr %row9.1, i64 %g9.1
  store i32 0, ptr %p9.1, align 4
  %row9.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g9.2
  %p9.2 = getelementptr inbounds nuw [4 x i8], ptr %row9.2, i64 %g9.2
  store i32 0, ptr %p9.2, align 4
  %row9.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g9.3
  %p9.3 = getelementptr inbounds nuw [4 x i8], ptr %row9.3, i64 %g9.3
  store i32 0, ptr %p9.3, align 4
  %row9.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g9.4
  %p9.4 = getelementptr inbounds nuw [4 x i8], ptr %row9.4, i64 %g9.4
  store i32 0, ptr %p9.4, align 4
  %row9.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g9.5
  %p9.5 = getelementptr inbounds nuw [4 x i8], ptr %row9.5, i64 %g9.5
  store i32 0, ptr %p9.5, align 4
  %row9.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g9.6
  %p9.6 = getelementptr inbounds nuw [4 x i8], ptr %row9.6, i64 %g9.6
  store i32 0, ptr %p9.6, align 4
  %row9.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g9.7
  %p9.7 = getelementptr inbounds nuw [4 x i8], ptr %row9.7, i64 %g9.7
  store i32 0, ptr %p9.7, align 4
  %i.10 = add nuw nsw i64 %i, 80
  %g10.1 = or disjoint i64 %i.10, 1
  %g10.2 = or disjoint i64 %i.10, 2
  %g10.3 = or disjoint i64 %i.10, 3
  %g10.4 = or disjoint i64 %i.10, 4
  %g10.5 = or disjoint i64 %i.10, 5
  %g10.6 = or disjoint i64 %i.10, 6
  %g10.7 = or disjoint i64 %i.10, 7
  %row10.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.10
  %p10.0 = getelementptr inbounds nuw [4 x i8], ptr %row10.0, i64 %i.10
  store i32 0, ptr %p10.0, align 4
  %row10.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g10.1
  %p10.1 = getelementptr inbounds nuw [4 x i8], ptr %row10.1, i64 %g10.1
  store i32 0, ptr %p10.1, align 4
  %row10.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g10.2
  %p10.2 = getelementptr inbounds nuw [4 x i8], ptr %row10.2, i64 %g10.2
  store i32 0, ptr %p10.2, align 4
  %row10.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g10.3
  %p10.3 = getelementptr inbounds nuw [4 x i8], ptr %row10.3, i64 %g10.3
  store i32 0, ptr %p10.3, align 4
  %row10.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g10.4
  %p10.4 = getelementptr inbounds nuw [4 x i8], ptr %row10.4, i64 %g10.4
  store i32 0, ptr %p10.4, align 4
  %row10.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g10.5
  %p10.5 = getelementptr inbounds nuw [4 x i8], ptr %row10.5, i64 %g10.5
  store i32 0, ptr %p10.5, align 4
  %row10.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g10.6
  %p10.6 = getelementptr inbounds nuw [4 x i8], ptr %row10.6, i64 %g10.6
  store i32 0, ptr %p10.6, align 4
  %row10.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g10.7
  %p10.7 = getelementptr inbounds nuw [4 x i8], ptr %row10.7, i64 %g10.7
  store i32 0, ptr %p10.7, align 4
  %i.11 = add nuw nsw i64 %i, 88
  %g11.1 = or disjoint i64 %i.11, 1
  %g11.2 = or disjoint i64 %i.11, 2
  %g11.3 = or disjoint i64 %i.11, 3
  %g11.4 = or disjoint i64 %i.11, 4
  %g11.5 = or disjoint i64 %i.11, 5
  %g11.6 = or disjoint i64 %i.11, 6
  %g11.7 = or disjoint i64 %i.11, 7
  %row11.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.11
  %p11.0 = getelementptr inbounds nuw [4 x i8], ptr %row11.0, i64 %i.11
  store i32 0, ptr %p11.0, align 4
  %row11.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g11.1
  %p11.1 = getelementptr inbounds nuw [4 x i8], ptr %row11.1, i64 %g11.1
  store i32 0, ptr %p11.1, align 4
  %row11.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g11.2
  %p11.2 = getelementptr inbounds nuw [4 x i8], ptr %row11.2, i64 %g11.2
  store i32 0, ptr %p11.2, align 4
  %row11.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g11.3
  %p11.3 = getelementptr inbounds nuw [4 x i8], ptr %row11.3, i64 %g11.3
  store i32 0, ptr %p11.3, align 4
  %row11.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g11.4
  %p11.4 = getelementptr inbounds nuw [4 x i8], ptr %row11.4, i64 %g11.4
  store i32 0, ptr %p11.4, align 4
  %row11.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g11.5
  %p11.5 = getelementptr inbounds nuw [4 x i8], ptr %row11.5, i64 %g11.5
  store i32 0, ptr %p11.5, align 4
  %row11.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g11.6
  %p11.6 = getelementptr inbounds nuw [4 x i8], ptr %row11.6, i64 %g11.6
  store i32 0, ptr %p11.6, align 4
  %row11.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g11.7
  %p11.7 = getelementptr inbounds nuw [4 x i8], ptr %row11.7, i64 %g11.7
  store i32 0, ptr %p11.7, align 4
  %i.12 = add nuw nsw i64 %i, 96
  %g12.1 = or disjoint i64 %i.12, 1
  %g12.2 = or disjoint i64 %i.12, 2
  %g12.3 = or disjoint i64 %i.12, 3
  %g12.4 = or disjoint i64 %i.12, 4
  %g12.5 = or disjoint i64 %i.12, 5
  %g12.6 = or disjoint i64 %i.12, 6
  %g12.7 = or disjoint i64 %i.12, 7
  %row12.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.12
  %p12.0 = getelementptr inbounds nuw [4 x i8], ptr %row12.0, i64 %i.12
  store i32 0, ptr %p12.0, align 4
  %row12.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g12.1
  %p12.1 = getelementptr inbounds nuw [4 x i8], ptr %row12.1, i64 %g12.1
  store i32 0, ptr %p12.1, align 4
  %row12.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g12.2
  %p12.2 = getelementptr inbounds nuw [4 x i8], ptr %row12.2, i64 %g12.2
  store i32 0, ptr %p12.2, align 4
  %row12.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g12.3
  %p12.3 = getelementptr inbounds nuw [4 x i8], ptr %row12.3, i64 %g12.3
  store i32 0, ptr %p12.3, align 4
  %row12.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g12.4
  %p12.4 = getelementptr inbounds nuw [4 x i8], ptr %row12.4, i64 %g12.4
  store i32 0, ptr %p12.4, align 4
  %row12.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g12.5
  %p12.5 = getelementptr inbounds nuw [4 x i8], ptr %row12.5, i64 %g12.5
  store i32 0, ptr %p12.5, align 4
  %row12.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g12.6
  %p12.6 = getelementptr inbounds nuw [4 x i8], ptr %row12.6, i64 %g12.6
  store i32 0, ptr %p12.6, align 4
  %row12.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g12.7
  %p12.7 = getelementptr inbounds nuw [4 x i8], ptr %row12.7, i64 %g12.7
  store i32 0, ptr %p12.7, align 4
  %i.13 = add nuw nsw i64 %i, 104
  %g13.1 = or disjoint i64 %i.13, 1
  %g13.2 = or disjoint i64 %i.13, 2
  %g13.3 = or disjoint i64 %i.13, 3
  %g13.4 = or disjoint i64 %i.13, 4
  %g13.5 = or disjoint i64 %i.13, 5
  %g13.6 = or disjoint i64 %i.13, 6
  %g13.7 = or disjoint i64 %i.13, 7
  %row13.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.13
  %p13.0 = getelementptr inbounds nuw [4 x i8], ptr %row13.0, i64 %i.13
  store i32 0, ptr %p13.0, align 4
  %row13.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g13.1
  %p13.1 = getelementptr inbounds nuw [4 x i8], ptr %row13.1, i64 %g13.1
  store i32 0, ptr %p13.1, align 4
  %row13.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g13.2
  %p13.2 = getelementptr inbounds nuw [4 x i8], ptr %row13.2, i64 %g13.2
  store i32 0, ptr %p13.2, align 4
  %row13.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g13.3
  %p13.3 = getelementptr inbounds nuw [4 x i8], ptr %row13.3, i64 %g13.3
  store i32 0, ptr %p13.3, align 4
  %row13.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g13.4
  %p13.4 = getelementptr inbounds nuw [4 x i8], ptr %row13.4, i64 %g13.4
  store i32 0, ptr %p13.4, align 4
  %row13.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g13.5
  %p13.5 = getelementptr inbounds nuw [4 x i8], ptr %row13.5, i64 %g13.5
  store i32 0, ptr %p13.5, align 4
  %row13.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g13.6
  %p13.6 = getelementptr inbounds nuw [4 x i8], ptr %row13.6, i64 %g13.6
  store i32 0, ptr %p13.6, align 4
  %row13.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g13.7
  %p13.7 = getelementptr inbounds nuw [4 x i8], ptr %row13.7, i64 %g13.7
  store i32 0, ptr %p13.7, align 4
  %i.14 = add nuw nsw i64 %i, 112
  %g14.1 = or disjoint i64 %i.14, 1
  %g14.2 = or disjoint i64 %i.14, 2
  %g14.3 = or disjoint i64 %i.14, 3
  %g14.4 = or disjoint i64 %i.14, 4
  %g14.5 = or disjoint i64 %i.14, 5
  %g14.6 = or disjoint i64 %i.14, 6
  %g14.7 = or disjoint i64 %i.14, 7
  %row14.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.14
  %p14.0 = getelementptr inbounds nuw [4 x i8], ptr %row14.0, i64 %i.14
  store i32 0, ptr %p14.0, align 4
  %row14.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g14.1
  %p14.1 = getelementptr inbounds nuw [4 x i8], ptr %row14.1, i64 %g14.1
  store i32 0, ptr %p14.1, align 4
  %row14.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g14.2
  %p14.2 = getelementptr inbounds nuw [4 x i8], ptr %row14.2, i64 %g14.2
  store i32 0, ptr %p14.2, align 4
  %row14.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g14.3
  %p14.3 = getelementptr inbounds nuw [4 x i8], ptr %row14.3, i64 %g14.3
  store i32 0, ptr %p14.3, align 4
  %row14.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g14.4
  %p14.4 = getelementptr inbounds nuw [4 x i8], ptr %row14.4, i64 %g14.4
  store i32 0, ptr %p14.4, align 4
  %row14.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g14.5
  %p14.5 = getelementptr inbounds nuw [4 x i8], ptr %row14.5, i64 %g14.5
  store i32 0, ptr %p14.5, align 4
  %row14.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g14.6
  %p14.6 = getelementptr inbounds nuw [4 x i8], ptr %row14.6, i64 %g14.6
  store i32 0, ptr %p14.6, align 4
  %row14.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g14.7
  %p14.7 = getelementptr inbounds nuw [4 x i8], ptr %row14.7, i64 %g14.7
  store i32 0, ptr %p14.7, align 4
  %i.15 = add nuw nsw i64 %i, 120
  %g15.1 = or disjoint i64 %i.15, 1
  %g15.2 = or disjoint i64 %i.15, 2
  %g15.3 = or disjoint i64 %i.15, 3
  %g15.4 = or disjoint i64 %i.15, 4
  %g15.5 = or disjoint i64 %i.15, 5
  %g15.6 = or disjoint i64 %i.15, 6
  %g15.7 = or disjoint i64 %i.15, 7
  %row15.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.15
  %p15.0 = getelementptr inbounds nuw [4 x i8], ptr %row15.0, i64 %i.15
  store i32 0, ptr %p15.0, align 4
  %row15.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g15.1
  %p15.1 = getelementptr inbounds nuw [4 x i8], ptr %row15.1, i64 %g15.1
  store i32 0, ptr %p15.1, align 4
  %row15.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g15.2
  %p15.2 = getelementptr inbounds nuw [4 x i8], ptr %row15.2, i64 %g15.2
  store i32 0, ptr %p15.2, align 4
  %row15.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g15.3
  %p15.3 = getelementptr inbounds nuw [4 x i8], ptr %row15.3, i64 %g15.3
  store i32 0, ptr %p15.3, align 4
  %row15.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g15.4
  %p15.4 = getelementptr inbounds nuw [4 x i8], ptr %row15.4, i64 %g15.4
  store i32 0, ptr %p15.4, align 4
  %row15.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g15.5
  %p15.5 = getelementptr inbounds nuw [4 x i8], ptr %row15.5, i64 %g15.5
  store i32 0, ptr %p15.5, align 4
  %row15.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g15.6
  %p15.6 = getelementptr inbounds nuw [4 x i8], ptr %row15.6, i64 %g15.6
  store i32 0, ptr %p15.6, align 4
  %row15.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g15.7
  %p15.7 = getelementptr inbounds nuw [4 x i8], ptr %row15.7, i64 %g15.7
  store i32 0, ptr %p15.7, align 4
  %i.16 = add nuw nsw i64 %i, 128
  %g16.1 = or disjoint i64 %i.16, 1
  %g16.2 = or disjoint i64 %i.16, 2
  %g16.3 = or disjoint i64 %i.16, 3
  %g16.4 = or disjoint i64 %i.16, 4
  %g16.5 = or disjoint i64 %i.16, 5
  %g16.6 = or disjoint i64 %i.16, 6
  %g16.7 = or disjoint i64 %i.16, 7
  %row16.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.16
  %p16.0 = getelementptr inbounds nuw [4 x i8], ptr %row16.0, i64 %i.16
  store i32 0, ptr %p16.0, align 4
  %row16.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g16.1
  %p16.1 = getelementptr inbounds nuw [4 x i8], ptr %row16.1, i64 %g16.1
  store i32 0, ptr %p16.1, align 4
  %row16.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g16.2
  %p16.2 = getelementptr inbounds nuw [4 x i8], ptr %row16.2, i64 %g16.2
  store i32 0, ptr %p16.2, align 4
  %row16.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g16.3
  %p16.3 = getelementptr inbounds nuw [4 x i8], ptr %row16.3, i64 %g16.3
  store i32 0, ptr %p16.3, align 4
  %row16.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g16.4
  %p16.4 = getelementptr inbounds nuw [4 x i8], ptr %row16.4, i64 %g16.4
  store i32 0, ptr %p16.4, align 4
  %row16.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g16.5
  %p16.5 = getelementptr inbounds nuw [4 x i8], ptr %row16.5, i64 %g16.5
  store i32 0, ptr %p16.5, align 4
  %row16.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g16.6
  %p16.6 = getelementptr inbounds nuw [4 x i8], ptr %row16.6, i64 %g16.6
  store i32 0, ptr %p16.6, align 4
  %row16.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g16.7
  %p16.7 = getelementptr inbounds nuw [4 x i8], ptr %row16.7, i64 %g16.7
  store i32 0, ptr %p16.7, align 4
  %i.17 = add nuw nsw i64 %i, 136
  %g17.1 = or disjoint i64 %i.17, 1
  %g17.2 = or disjoint i64 %i.17, 2
  %g17.3 = or disjoint i64 %i.17, 3
  %g17.4 = or disjoint i64 %i.17, 4
  %g17.5 = or disjoint i64 %i.17, 5
  %g17.6 = or disjoint i64 %i.17, 6
  %g17.7 = or disjoint i64 %i.17, 7
  %row17.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.17
  %p17.0 = getelementptr inbounds nuw [4 x i8], ptr %row17.0, i64 %i.17
  store i32 0, ptr %p17.0, align 4
  %row17.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g17.1
  %p17.1 = getelementptr inbounds nuw [4 x i8], ptr %row17.1, i64 %g17.1
  store i32 0, ptr %p17.1, align 4
  %row17.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g17.2
  %p17.2 = getelementptr inbounds nuw [4 x i8], ptr %row17.2, i64 %g17.2
  store i32 0, ptr %p17.2, align 4
  %row17.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g17.3
  %p17.3 = getelementptr inbounds nuw [4 x i8], ptr %row17.3, i64 %g17.3
  store i32 0, ptr %p17.3, align 4
  %row17.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g17.4
  %p17.4 = getelementptr inbounds nuw [4 x i8], ptr %row17.4, i64 %g17.4
  store i32 0, ptr %p17.4, align 4
  %row17.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g17.5
  %p17.5 = getelementptr inbounds nuw [4 x i8], ptr %row17.5, i64 %g17.5
  store i32 0, ptr %p17.5, align 4
  %row17.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g17.6
  %p17.6 = getelementptr inbounds nuw [4 x i8], ptr %row17.6, i64 %g17.6
  store i32 0, ptr %p17.6, align 4
  %row17.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g17.7
  %p17.7 = getelementptr inbounds nuw [4 x i8], ptr %row17.7, i64 %g17.7
  store i32 0, ptr %p17.7, align 4
  %i.18 = add nuw nsw i64 %i, 144
  %g18.1 = or disjoint i64 %i.18, 1
  %g18.2 = or disjoint i64 %i.18, 2
  %g18.3 = or disjoint i64 %i.18, 3
  %g18.4 = or disjoint i64 %i.18, 4
  %g18.5 = or disjoint i64 %i.18, 5
  %g18.6 = or disjoint i64 %i.18, 6
  %g18.7 = or disjoint i64 %i.18, 7
  %row18.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.18
  %p18.0 = getelementptr inbounds nuw [4 x i8], ptr %row18.0, i64 %i.18
  store i32 0, ptr %p18.0, align 4
  %row18.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g18.1
  %p18.1 = getelementptr inbounds nuw [4 x i8], ptr %row18.1, i64 %g18.1
  store i32 0, ptr %p18.1, align 4
  %row18.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g18.2
  %p18.2 = getelementptr inbounds nuw [4 x i8], ptr %row18.2, i64 %g18.2
  store i32 0, ptr %p18.2, align 4
  %row18.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g18.3
  %p18.3 = getelementptr inbounds nuw [4 x i8], ptr %row18.3, i64 %g18.3
  store i32 0, ptr %p18.3, align 4
  %row18.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g18.4
  %p18.4 = getelementptr inbounds nuw [4 x i8], ptr %row18.4, i64 %g18.4
  store i32 0, ptr %p18.4, align 4
  %row18.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g18.5
  %p18.5 = getelementptr inbounds nuw [4 x i8], ptr %row18.5, i64 %g18.5
  store i32 0, ptr %p18.5, align 4
  %row18.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g18.6
  %p18.6 = getelementptr inbounds nuw [4 x i8], ptr %row18.6, i64 %g18.6
  store i32 0, ptr %p18.6, align 4
  %row18.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g18.7
  %p18.7 = getelementptr inbounds nuw [4 x i8], ptr %row18.7, i64 %g18.7
  store i32 0, ptr %p18.7, align 4
  %i.19 = add nuw nsw i64 %i, 152
  %g19.1 = or disjoint i64 %i.19, 1
  %g19.2 = or disjoint i64 %i.19, 2
  %g19.3 = or disjoint i64 %i.19, 3
  %g19.4 = or disjoint i64 %i.19, 4
  %g19.5 = or disjoint i64 %i.19, 5
  %g19.6 = or disjoint i64 %i.19, 6
  %g19.7 = or disjoint i64 %i.19, 7
  %row19.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.19
  %p19.0 = getelementptr inbounds nuw [4 x i8], ptr %row19.0, i64 %i.19
  store i32 0, ptr %p19.0, align 4
  %row19.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g19.1
  %p19.1 = getelementptr inbounds nuw [4 x i8], ptr %row19.1, i64 %g19.1
  store i32 0, ptr %p19.1, align 4
  %row19.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g19.2
  %p19.2 = getelementptr inbounds nuw [4 x i8], ptr %row19.2, i64 %g19.2
  store i32 0, ptr %p19.2, align 4
  %row19.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g19.3
  %p19.3 = getelementptr inbounds nuw [4 x i8], ptr %row19.3, i64 %g19.3
  store i32 0, ptr %p19.3, align 4
  %row19.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g19.4
  %p19.4 = getelementptr inbounds nuw [4 x i8], ptr %row19.4, i64 %g19.4
  store i32 0, ptr %p19.4, align 4
  %row19.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g19.5
  %p19.5 = getelementptr inbounds nuw [4 x i8], ptr %row19.5, i64 %g19.5
  store i32 0, ptr %p19.5, align 4
  %row19.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g19.6
  %p19.6 = getelementptr inbounds nuw [4 x i8], ptr %row19.6, i64 %g19.6
  store i32 0, ptr %p19.6, align 4
  %row19.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g19.7
  %p19.7 = getelementptr inbounds nuw [4 x i8], ptr %row19.7, i64 %g19.7
  store i32 0, ptr %p19.7, align 4
  %i.20 = add nuw nsw i64 %i, 160
  %g20.1 = or disjoint i64 %i.20, 1
  %g20.2 = or disjoint i64 %i.20, 2
  %g20.3 = or disjoint i64 %i.20, 3
  %g20.4 = or disjoint i64 %i.20, 4
  %g20.5 = or disjoint i64 %i.20, 5
  %g20.6 = or disjoint i64 %i.20, 6
  %g20.7 = or disjoint i64 %i.20, 7
  %row20.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.20
  %p20.0 = getelementptr inbounds nuw [4 x i8], ptr %row20.0, i64 %i.20
  store i32 0, ptr %p20.0, align 4
  %row20.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g20.1
  %p20.1 = getelementptr inbounds nuw [4 x i8], ptr %row20.1, i64 %g20.1
  store i32 0, ptr %p20.1, align 4
  %row20.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g20.2
  %p20.2 = getelementptr inbounds nuw [4 x i8], ptr %row20.2, i64 %g20.2
  store i32 0, ptr %p20.2, align 4
  %row20.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g20.3
  %p20.3 = getelementptr inbounds nuw [4 x i8], ptr %row20.3, i64 %g20.3
  store i32 0, ptr %p20.3, align 4
  %row20.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g20.4
  %p20.4 = getelementptr inbounds nuw [4 x i8], ptr %row20.4, i64 %g20.4
  store i32 0, ptr %p20.4, align 4
  %row20.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g20.5
  %p20.5 = getelementptr inbounds nuw [4 x i8], ptr %row20.5, i64 %g20.5
  store i32 0, ptr %p20.5, align 4
  %row20.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g20.6
  %p20.6 = getelementptr inbounds nuw [4 x i8], ptr %row20.6, i64 %g20.6
  store i32 0, ptr %p20.6, align 4
  %row20.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g20.7
  %p20.7 = getelementptr inbounds nuw [4 x i8], ptr %row20.7, i64 %g20.7
  store i32 0, ptr %p20.7, align 4
  %i.21 = add nuw nsw i64 %i, 168
  %g21.1 = or disjoint i64 %i.21, 1
  %g21.2 = or disjoint i64 %i.21, 2
  %g21.3 = or disjoint i64 %i.21, 3
  %g21.4 = or disjoint i64 %i.21, 4
  %g21.5 = or disjoint i64 %i.21, 5
  %g21.6 = or disjoint i64 %i.21, 6
  %g21.7 = or disjoint i64 %i.21, 7
  %row21.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.21
  %p21.0 = getelementptr inbounds nuw [4 x i8], ptr %row21.0, i64 %i.21
  store i32 0, ptr %p21.0, align 4
  %row21.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g21.1
  %p21.1 = getelementptr inbounds nuw [4 x i8], ptr %row21.1, i64 %g21.1
  store i32 0, ptr %p21.1, align 4
  %row21.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g21.2
  %p21.2 = getelementptr inbounds nuw [4 x i8], ptr %row21.2, i64 %g21.2
  store i32 0, ptr %p21.2, align 4
  %row21.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g21.3
  %p21.3 = getelementptr inbounds nuw [4 x i8], ptr %row21.3, i64 %g21.3
  store i32 0, ptr %p21.3, align 4
  %row21.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g21.4
  %p21.4 = getelementptr inbounds nuw [4 x i8], ptr %row21.4, i64 %g21.4
  store i32 0, ptr %p21.4, align 4
  %row21.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g21.5
  %p21.5 = getelementptr inbounds nuw [4 x i8], ptr %row21.5, i64 %g21.5
  store i32 0, ptr %p21.5, align 4
  %row21.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g21.6
  %p21.6 = getelementptr inbounds nuw [4 x i8], ptr %row21.6, i64 %g21.6
  store i32 0, ptr %p21.6, align 4
  %row21.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g21.7
  %p21.7 = getelementptr inbounds nuw [4 x i8], ptr %row21.7, i64 %g21.7
  store i32 0, ptr %p21.7, align 4
  %i.22 = add nuw nsw i64 %i, 176
  %g22.1 = or disjoint i64 %i.22, 1
  %g22.2 = or disjoint i64 %i.22, 2
  %g22.3 = or disjoint i64 %i.22, 3
  %g22.4 = or disjoint i64 %i.22, 4
  %g22.5 = or disjoint i64 %i.22, 5
  %g22.6 = or disjoint i64 %i.22, 6
  %g22.7 = or disjoint i64 %i.22, 7
  %row22.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.22
  %p22.0 = getelementptr inbounds nuw [4 x i8], ptr %row22.0, i64 %i.22
  store i32 0, ptr %p22.0, align 4
  %row22.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g22.1
  %p22.1 = getelementptr inbounds nuw [4 x i8], ptr %row22.1, i64 %g22.1
  store i32 0, ptr %p22.1, align 4
  %row22.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g22.2
  %p22.2 = getelementptr inbounds nuw [4 x i8], ptr %row22.2, i64 %g22.2
  store i32 0, ptr %p22.2, align 4
  %row22.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g22.3
  %p22.3 = getelementptr inbounds nuw [4 x i8], ptr %row22.3, i64 %g22.3
  store i32 0, ptr %p22.3, align 4
  %row22.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g22.4
  %p22.4 = getelementptr inbounds nuw [4 x i8], ptr %row22.4, i64 %g22.4
  store i32 0, ptr %p22.4, align 4
  %row22.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g22.5
  %p22.5 = getelementptr inbounds nuw [4 x i8], ptr %row22.5, i64 %g22.5
  store i32 0, ptr %p22.5, align 4
  %row22.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g22.6
  %p22.6 = getelementptr inbounds nuw [4 x i8], ptr %row22.6, i64 %g22.6
  store i32 0, ptr %p22.6, align 4
  %row22.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g22.7
  %p22.7 = getelementptr inbounds nuw [4 x i8], ptr %row22.7, i64 %g22.7
  store i32 0, ptr %p22.7, align 4
  %i.23 = add nuw nsw i64 %i, 184
  %g23.1 = or disjoint i64 %i.23, 1
  %g23.2 = or disjoint i64 %i.23, 2
  %g23.3 = or disjoint i64 %i.23, 3
  %g23.4 = or disjoint i64 %i.23, 4
  %g23.5 = or disjoint i64 %i.23, 5
  %g23.6 = or disjoint i64 %i.23, 6
  %g23.7 = or disjoint i64 %i.23, 7
  %row23.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.23
  %p23.0 = getelementptr inbounds nuw [4 x i8], ptr %row23.0, i64 %i.23
  store i32 0, ptr %p23.0, align 4
  %row23.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g23.1
  %p23.1 = getelementptr inbounds nuw [4 x i8], ptr %row23.1, i64 %g23.1
  store i32 0, ptr %p23.1, align 4
  %row23.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g23.2
  %p23.2 = getelementptr inbounds nuw [4 x i8], ptr %row23.2, i64 %g23.2
  store i32 0, ptr %p23.2, align 4
  %row23.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g23.3
  %p23.3 = getelementptr inbounds nuw [4 x i8], ptr %row23.3, i64 %g23.3
  store i32 0, ptr %p23.3, align 4
  %row23.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g23.4
  %p23.4 = getelementptr inbounds nuw [4 x i8], ptr %row23.4, i64 %g23.4
  store i32 0, ptr %p23.4, align 4
  %row23.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g23.5
  %p23.5 = getelementptr inbounds nuw [4 x i8], ptr %row23.5, i64 %g23.5
  store i32 0, ptr %p23.5, align 4
  %row23.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g23.6
  %p23.6 = getelementptr inbounds nuw [4 x i8], ptr %row23.6, i64 %g23.6
  store i32 0, ptr %p23.6, align 4
  %row23.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g23.7
  %p23.7 = getelementptr inbounds nuw [4 x i8], ptr %row23.7, i64 %g23.7
  store i32 0, ptr %p23.7, align 4
  %i.24 = add nuw nsw i64 %i, 192
  %g24.1 = or disjoint i64 %i.24, 1
  %g24.2 = or disjoint i64 %i.24, 2
  %g24.3 = or disjoint i64 %i.24, 3
  %g24.4 = or disjoint i64 %i.24, 4
  %g24.5 = or disjoint i64 %i.24, 5
  %g24.6 = or disjoint i64 %i.24, 6
  %g24.7 = or disjoint i64 %i.24, 7
  %row24.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.24
  %p24.0 = getelementptr inbounds nuw [4 x i8], ptr %row24.0, i64 %i.24
  store i32 0, ptr %p24.0, align 4
  %row24.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g24.1
  %p24.1 = getelementptr inbounds nuw [4 x i8], ptr %row24.1, i64 %g24.1
  store i32 0, ptr %p24.1, align 4
  %row24.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g24.2
  %p24.2 = getelementptr inbounds nuw [4 x i8], ptr %row24.2, i64 %g24.2
  store i32 0, ptr %p24.2, align 4
  %row24.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g24.3
  %p24.3 = getelementptr inbounds nuw [4 x i8], ptr %row24.3, i64 %g24.3
  store i32 0, ptr %p24.3, align 4
  %row24.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g24.4
  %p24.4 = getelementptr inbounds nuw [4 x i8], ptr %row24.4, i64 %g24.4
  store i32 0, ptr %p24.4, align 4
  %row24.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g24.5
  %p24.5 = getelementptr inbounds nuw [4 x i8], ptr %row24.5, i64 %g24.5
  store i32 0, ptr %p24.5, align 4
  %row24.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g24.6
  %p24.6 = getelementptr inbounds nuw [4 x i8], ptr %row24.6, i64 %g24.6
  store i32 0, ptr %p24.6, align 4
  %row24.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g24.7
  %p24.7 = getelementptr inbounds nuw [4 x i8], ptr %row24.7, i64 %g24.7
  store i32 0, ptr %p24.7, align 4
  %i.25 = add nuw nsw i64 %i, 200
  %g25.1 = or disjoint i64 %i.25, 1
  %g25.2 = or disjoint i64 %i.25, 2
  %g25.3 = or disjoint i64 %i.25, 3
  %g25.4 = or disjoint i64 %i.25, 4
  %g25.5 = or disjoint i64 %i.25, 5
  %g25.6 = or disjoint i64 %i.25, 6
  %g25.7 = or disjoint i64 %i.25, 7
  %row25.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.25
  %p25.0 = getelementptr inbounds nuw [4 x i8], ptr %row25.0, i64 %i.25
  store i32 0, ptr %p25.0, align 4
  %row25.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g25.1
  %p25.1 = getelementptr inbounds nuw [4 x i8], ptr %row25.1, i64 %g25.1
  store i32 0, ptr %p25.1, align 4
  %row25.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g25.2
  %p25.2 = getelementptr inbounds nuw [4 x i8], ptr %row25.2, i64 %g25.2
  store i32 0, ptr %p25.2, align 4
  %row25.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g25.3
  %p25.3 = getelementptr inbounds nuw [4 x i8], ptr %row25.3, i64 %g25.3
  store i32 0, ptr %p25.3, align 4
  %row25.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g25.4
  %p25.4 = getelementptr inbounds nuw [4 x i8], ptr %row25.4, i64 %g25.4
  store i32 0, ptr %p25.4, align 4
  %row25.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g25.5
  %p25.5 = getelementptr inbounds nuw [4 x i8], ptr %row25.5, i64 %g25.5
  store i32 0, ptr %p25.5, align 4
  %row25.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g25.6
  %p25.6 = getelementptr inbounds nuw [4 x i8], ptr %row25.6, i64 %g25.6
  store i32 0, ptr %p25.6, align 4
  %row25.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g25.7
  %p25.7 = getelementptr inbounds nuw [4 x i8], ptr %row25.7, i64 %g25.7
  store i32 0, ptr %p25.7, align 4
  %i.26 = add nuw nsw i64 %i, 208
  %g26.1 = or disjoint i64 %i.26, 1
  %g26.2 = or disjoint i64 %i.26, 2
  %g26.3 = or disjoint i64 %i.26, 3
  %g26.4 = or disjoint i64 %i.26, 4
  %g26.5 = or disjoint i64 %i.26, 5
  %g26.6 = or disjoint i64 %i.26, 6
  %g26.7 = or disjoint i64 %i.26, 7
  %row26.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.26
  %p26.0 = getelementptr inbounds nuw [4 x i8], ptr %row26.0, i64 %i.26
  store i32 0, ptr %p26.0, align 4
  %row26.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g26.1
  %p26.1 = getelementptr inbounds nuw [4 x i8], ptr %row26.1, i64 %g26.1
  store i32 0, ptr %p26.1, align 4
  %row26.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g26.2
  %p26.2 = getelementptr inbounds nuw [4 x i8], ptr %row26.2, i64 %g26.2
  store i32 0, ptr %p26.2, align 4
  %row26.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g26.3
  %p26.3 = getelementptr inbounds nuw [4 x i8], ptr %row26.3, i64 %g26.3
  store i32 0, ptr %p26.3, align 4
  %row26.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g26.4
  %p26.4 = getelementptr inbounds nuw [4 x i8], ptr %row26.4, i64 %g26.4
  store i32 0, ptr %p26.4, align 4
  %row26.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g26.5
  %p26.5 = getelementptr inbounds nuw [4 x i8], ptr %row26.5, i64 %g26.5
  store i32 0, ptr %p26.5, align 4
  %row26.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g26.6
  %p26.6 = getelementptr inbounds nuw [4 x i8], ptr %row26.6, i64 %g26.6
  store i32 0, ptr %p26.6, align 4
  %row26.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g26.7
  %p26.7 = getelementptr inbounds nuw [4 x i8], ptr %row26.7, i64 %g26.7
  store i32 0, ptr %p26.7, align 4
  %i.27 = add nuw nsw i64 %i, 216
  %g27.1 = or disjoint i64 %i.27, 1
  %g27.2 = or disjoint i64 %i.27, 2
  %g27.3 = or disjoint i64 %i.27, 3
  %g27.4 = or disjoint i64 %i.27, 4
  %g27.5 = or disjoint i64 %i.27, 5
  %g27.6 = or disjoint i64 %i.27, 6
  %g27.7 = or disjoint i64 %i.27, 7
  %row27.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.27
  %p27.0 = getelementptr inbounds nuw [4 x i8], ptr %row27.0, i64 %i.27
  store i32 0, ptr %p27.0, align 4
  %row27.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g27.1
  %p27.1 = getelementptr inbounds nuw [4 x i8], ptr %row27.1, i64 %g27.1
  store i32 0, ptr %p27.1, align 4
  %row27.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g27.2
  %p27.2 = getelementptr inbounds nuw [4 x i8], ptr %row27.2, i64 %g27.2
  store i32 0, ptr %p27.2, align 4
  %row27.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g27.3
  %p27.3 = getelementptr inbounds nuw [4 x i8], ptr %row27.3, i64 %g27.3
  store i32 0, ptr %p27.3, align 4
  %row27.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g27.4
  %p27.4 = getelementptr inbounds nuw [4 x i8], ptr %row27.4, i64 %g27.4
  store i32 0, ptr %p27.4, align 4
  %row27.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g27.5
  %p27.5 = getelementptr inbounds nuw [4 x i8], ptr %row27.5, i64 %g27.5
  store i32 0, ptr %p27.5, align 4
  %row27.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g27.6
  %p27.6 = getelementptr inbounds nuw [4 x i8], ptr %row27.6, i64 %g27.6
  store i32 0, ptr %p27.6, align 4
  %row27.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g27.7
  %p27.7 = getelementptr inbounds nuw [4 x i8], ptr %row27.7, i64 %g27.7
  store i32 0, ptr %p27.7, align 4
  %i.28 = add nuw nsw i64 %i, 224
  %g28.1 = or disjoint i64 %i.28, 1
  %g28.2 = or disjoint i64 %i.28, 2
  %g28.3 = or disjoint i64 %i.28, 3
  %g28.4 = or disjoint i64 %i.28, 4
  %g28.5 = or disjoint i64 %i.28, 5
  %g28.6 = or disjoint i64 %i.28, 6
  %g28.7 = or disjoint i64 %i.28, 7
  %row28.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.28
  %p28.0 = getelementptr inbounds nuw [4 x i8], ptr %row28.0, i64 %i.28
  store i32 0, ptr %p28.0, align 4
  %row28.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g28.1
  %p28.1 = getelementptr inbounds nuw [4 x i8], ptr %row28.1, i64 %g28.1
  store i32 0, ptr %p28.1, align 4
  %row28.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g28.2
  %p28.2 = getelementptr inbounds nuw [4 x i8], ptr %row28.2, i64 %g28.2
  store i32 0, ptr %p28.2, align 4
  %row28.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g28.3
  %p28.3 = getelementptr inbounds nuw [4 x i8], ptr %row28.3, i64 %g28.3
  store i32 0, ptr %p28.3, align 4
  %row28.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g28.4
  %p28.4 = getelementptr inbounds nuw [4 x i8], ptr %row28.4, i64 %g28.4
  store i32 0, ptr %p28.4, align 4
  %row28.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g28.5
  %p28.5 = getelementptr inbounds nuw [4 x i8], ptr %row28.5, i64 %g28.5
  store i32 0, ptr %p28.5, align 4
  %row28.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g28.6
  %p28.6 = getelementptr inbounds nuw [4 x i8], ptr %row28.6, i64 %g28.6
  store i32 0, ptr %p28.6, align 4
  %row28.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g28.7
  %p28.7 = getelementptr inbounds nuw [4 x i8], ptr %row28.7, i64 %g28.7
  store i32 0, ptr %p28.7, align 4
  %i.29 = add nuw nsw i64 %i, 232
  %g29.1 = or disjoint i64 %i.29, 1
  %g29.2 = or disjoint i64 %i.29, 2
  %g29.3 = or disjoint i64 %i.29, 3
  %g29.4 = or disjoint i64 %i.29, 4
  %g29.5 = or disjoint i64 %i.29, 5
  %g29.6 = or disjoint i64 %i.29, 6
  %g29.7 = or disjoint i64 %i.29, 7
  %row29.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.29
  %p29.0 = getelementptr inbounds nuw [4 x i8], ptr %row29.0, i64 %i.29
  store i32 0, ptr %p29.0, align 4
  %row29.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g29.1
  %p29.1 = getelementptr inbounds nuw [4 x i8], ptr %row29.1, i64 %g29.1
  store i32 0, ptr %p29.1, align 4
  %row29.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g29.2
  %p29.2 = getelementptr inbounds nuw [4 x i8], ptr %row29.2, i64 %g29.2
  store i32 0, ptr %p29.2, align 4
  %row29.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g29.3
  %p29.3 = getelementptr inbounds nuw [4 x i8], ptr %row29.3, i64 %g29.3
  store i32 0, ptr %p29.3, align 4
  %row29.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g29.4
  %p29.4 = getelementptr inbounds nuw [4 x i8], ptr %row29.4, i64 %g29.4
  store i32 0, ptr %p29.4, align 4
  %row29.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g29.5
  %p29.5 = getelementptr inbounds nuw [4 x i8], ptr %row29.5, i64 %g29.5
  store i32 0, ptr %p29.5, align 4
  %row29.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g29.6
  %p29.6 = getelementptr inbounds nuw [4 x i8], ptr %row29.6, i64 %g29.6
  store i32 0, ptr %p29.6, align 4
  %row29.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g29.7
  %p29.7 = getelementptr inbounds nuw [4 x i8], ptr %row29.7, i64 %g29.7
  store i32 0, ptr %p29.7, align 4
  %i.30 = add nuw nsw i64 %i, 240
  %g30.1 = or disjoint i64 %i.30, 1
  %g30.2 = or disjoint i64 %i.30, 2
  %g30.3 = or disjoint i64 %i.30, 3
  %g30.4 = or disjoint i64 %i.30, 4
  %g30.5 = or disjoint i64 %i.30, 5
  %g30.6 = or disjoint i64 %i.30, 6
  %g30.7 = or disjoint i64 %i.30, 7
  %row30.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.30
  %p30.0 = getelementptr inbounds nuw [4 x i8], ptr %row30.0, i64 %i.30
  store i32 0, ptr %p30.0, align 4
  %row30.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g30.1
  %p30.1 = getelementptr inbounds nuw [4 x i8], ptr %row30.1, i64 %g30.1
  store i32 0, ptr %p30.1, align 4
  %row30.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g30.2
  %p30.2 = getelementptr inbounds nuw [4 x i8], ptr %row30.2, i64 %g30.2
  store i32 0, ptr %p30.2, align 4
  %row30.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g30.3
  %p30.3 = getelementptr inbounds nuw [4 x i8], ptr %row30.3, i64 %g30.3
  store i32 0, ptr %p30.3, align 4
  %row30.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g30.4
  %p30.4 = getelementptr inbounds nuw [4 x i8], ptr %row30.4, i64 %g30.4
  store i32 0, ptr %p30.4, align 4
  %row30.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g30.5
  %p30.5 = getelementptr inbounds nuw [4 x i8], ptr %row30.5, i64 %g30.5
  store i32 0, ptr %p30.5, align 4
  %row30.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g30.6
  %p30.6 = getelementptr inbounds nuw [4 x i8], ptr %row30.6, i64 %g30.6
  store i32 0, ptr %p30.6, align 4
  %row30.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g30.7
  %p30.7 = getelementptr inbounds nuw [4 x i8], ptr %row30.7, i64 %g30.7
  store i32 0, ptr %p30.7, align 4
  %i.31 = add nuw nsw i64 %i, 248
  %g31.1 = or disjoint i64 %i.31, 1
  %g31.2 = or disjoint i64 %i.31, 2
  %g31.3 = or disjoint i64 %i.31, 3
  %g31.4 = or disjoint i64 %i.31, 4
  %g31.5 = or disjoint i64 %i.31, 5
  %g31.6 = or disjoint i64 %i.31, 6
  %g31.7 = or disjoint i64 %i.31, 7
  %row31.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.31
  %p31.0 = getelementptr inbounds nuw [4 x i8], ptr %row31.0, i64 %i.31
  store i32 0, ptr %p31.0, align 4
  %row31.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g31.1
  %p31.1 = getelementptr inbounds nuw [4 x i8], ptr %row31.1, i64 %g31.1
  store i32 0, ptr %p31.1, align 4
  %row31.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g31.2
  %p31.2 = getelementptr inbounds nuw [4 x i8], ptr %row31.2, i64 %g31.2
  store i32 0, ptr %p31.2, align 4
  %row31.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g31.3
  %p31.3 = getelementptr inbounds nuw [4 x i8], ptr %row31.3, i64 %g31.3
  store i32 0, ptr %p31.3, align 4
  %row31.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g31.4
  %p31.4 = getelementptr inbounds nuw [4 x i8], ptr %row31.4, i64 %g31.4
  store i32 0, ptr %p31.4, align 4
  %row31.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g31.5
  %p31.5 = getelementptr inbounds nuw [4 x i8], ptr %row31.5, i64 %g31.5
  store i32 0, ptr %p31.5, align 4
  %row31.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g31.6
  %p31.6 = getelementptr inbounds nuw [4 x i8], ptr %row31.6, i64 %g31.6
  store i32 0, ptr %p31.6, align 4
  %row31.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g31.7
  %p31.7 = getelementptr inbounds nuw [4 x i8], ptr %row31.7, i64 %g31.7
  store i32 0, ptr %p31.7, align 4
  %i.32 = add nuw nsw i64 %i, 256
  %g32.1 = or disjoint i64 %i.32, 1
  %g32.2 = or disjoint i64 %i.32, 2
  %g32.3 = or disjoint i64 %i.32, 3
  %g32.4 = or disjoint i64 %i.32, 4
  %g32.5 = or disjoint i64 %i.32, 5
  %g32.6 = or disjoint i64 %i.32, 6
  %g32.7 = or disjoint i64 %i.32, 7
  %row32.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.32
  %p32.0 = getelementptr inbounds nuw [4 x i8], ptr %row32.0, i64 %i.32
  store i32 0, ptr %p32.0, align 4
  %row32.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g32.1
  %p32.1 = getelementptr inbounds nuw [4 x i8], ptr %row32.1, i64 %g32.1
  store i32 0, ptr %p32.1, align 4
  %row32.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g32.2
  %p32.2 = getelementptr inbounds nuw [4 x i8], ptr %row32.2, i64 %g32.2
  store i32 0, ptr %p32.2, align 4
  %row32.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g32.3
  %p32.3 = getelementptr inbounds nuw [4 x i8], ptr %row32.3, i64 %g32.3
  store i32 0, ptr %p32.3, align 4
  %row32.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g32.4
  %p32.4 = getelementptr inbounds nuw [4 x i8], ptr %row32.4, i64 %g32.4
  store i32 0, ptr %p32.4, align 4
  %row32.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g32.5
  %p32.5 = getelementptr inbounds nuw [4 x i8], ptr %row32.5, i64 %g32.5
  store i32 0, ptr %p32.5, align 4
  %row32.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g32.6
  %p32.6 = getelementptr inbounds nuw [4 x i8], ptr %row32.6, i64 %g32.6
  store i32 0, ptr %p32.6, align 4
  %row32.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g32.7
  %p32.7 = getelementptr inbounds nuw [4 x i8], ptr %row32.7, i64 %g32.7
  store i32 0, ptr %p32.7, align 4
  %i.33 = add nuw nsw i64 %i, 264
  %g33.1 = or disjoint i64 %i.33, 1
  %g33.2 = or disjoint i64 %i.33, 2
  %g33.3 = or disjoint i64 %i.33, 3
  %g33.4 = or disjoint i64 %i.33, 4
  %g33.5 = or disjoint i64 %i.33, 5
  %g33.6 = or disjoint i64 %i.33, 6
  %g33.7 = or disjoint i64 %i.33, 7
  %row33.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.33
  %p33.0 = getelementptr inbounds nuw [4 x i8], ptr %row33.0, i64 %i.33
  store i32 0, ptr %p33.0, align 4
  %row33.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g33.1
  %p33.1 = getelementptr inbounds nuw [4 x i8], ptr %row33.1, i64 %g33.1
  store i32 0, ptr %p33.1, align 4
  %row33.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g33.2
  %p33.2 = getelementptr inbounds nuw [4 x i8], ptr %row33.2, i64 %g33.2
  store i32 0, ptr %p33.2, align 4
  %row33.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g33.3
  %p33.3 = getelementptr inbounds nuw [4 x i8], ptr %row33.3, i64 %g33.3
  store i32 0, ptr %p33.3, align 4
  %row33.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g33.4
  %p33.4 = getelementptr inbounds nuw [4 x i8], ptr %row33.4, i64 %g33.4
  store i32 0, ptr %p33.4, align 4
  %row33.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g33.5
  %p33.5 = getelementptr inbounds nuw [4 x i8], ptr %row33.5, i64 %g33.5
  store i32 0, ptr %p33.5, align 4
  %row33.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g33.6
  %p33.6 = getelementptr inbounds nuw [4 x i8], ptr %row33.6, i64 %g33.6
  store i32 0, ptr %p33.6, align 4
  %row33.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g33.7
  %p33.7 = getelementptr inbounds nuw [4 x i8], ptr %row33.7, i64 %g33.7
  store i32 0, ptr %p33.7, align 4
  %i.34 = add nuw nsw i64 %i, 272
  %g34.1 = or disjoint i64 %i.34, 1
  %g34.2 = or disjoint i64 %i.34, 2
  %g34.3 = or disjoint i64 %i.34, 3
  %g34.4 = or disjoint i64 %i.34, 4
  %g34.5 = or disjoint i64 %i.34, 5
  %g34.6 = or disjoint i64 %i.34, 6
  %g34.7 = or disjoint i64 %i.34, 7
  %row34.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.34
  %p34.0 = getelementptr inbounds nuw [4 x i8], ptr %row34.0, i64 %i.34
  store i32 0, ptr %p34.0, align 4
  %row34.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g34.1
  %p34.1 = getelementptr inbounds nuw [4 x i8], ptr %row34.1, i64 %g34.1
  store i32 0, ptr %p34.1, align 4
  %row34.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g34.2
  %p34.2 = getelementptr inbounds nuw [4 x i8], ptr %row34.2, i64 %g34.2
  store i32 0, ptr %p34.2, align 4
  %row34.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g34.3
  %p34.3 = getelementptr inbounds nuw [4 x i8], ptr %row34.3, i64 %g34.3
  store i32 0, ptr %p34.3, align 4
  %row34.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g34.4
  %p34.4 = getelementptr inbounds nuw [4 x i8], ptr %row34.4, i64 %g34.4
  store i32 0, ptr %p34.4, align 4
  %row34.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g34.5
  %p34.5 = getelementptr inbounds nuw [4 x i8], ptr %row34.5, i64 %g34.5
  store i32 0, ptr %p34.5, align 4
  %row34.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g34.6
  %p34.6 = getelementptr inbounds nuw [4 x i8], ptr %row34.6, i64 %g34.6
  store i32 0, ptr %p34.6, align 4
  %row34.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g34.7
  %p34.7 = getelementptr inbounds nuw [4 x i8], ptr %row34.7, i64 %g34.7
  store i32 0, ptr %p34.7, align 4
  %i.35 = add nuw nsw i64 %i, 280
  %g35.1 = or disjoint i64 %i.35, 1
  %g35.2 = or disjoint i64 %i.35, 2
  %g35.3 = or disjoint i64 %i.35, 3
  %g35.4 = or disjoint i64 %i.35, 4
  %g35.5 = or disjoint i64 %i.35, 5
  %g35.6 = or disjoint i64 %i.35, 6
  %g35.7 = or disjoint i64 %i.35, 7
  %row35.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.35
  %p35.0 = getelementptr inbounds nuw [4 x i8], ptr %row35.0, i64 %i.35
  store i32 0, ptr %p35.0, align 4
  %row35.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g35.1
  %p35.1 = getelementptr inbounds nuw [4 x i8], ptr %row35.1, i64 %g35.1
  store i32 0, ptr %p35.1, align 4
  %row35.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g35.2
  %p35.2 = getelementptr inbounds nuw [4 x i8], ptr %row35.2, i64 %g35.2
  store i32 0, ptr %p35.2, align 4
  %row35.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g35.3
  %p35.3 = getelementptr inbounds nuw [4 x i8], ptr %row35.3, i64 %g35.3
  store i32 0, ptr %p35.3, align 4
  %row35.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g35.4
  %p35.4 = getelementptr inbounds nuw [4 x i8], ptr %row35.4, i64 %g35.4
  store i32 0, ptr %p35.4, align 4
  %row35.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g35.5
  %p35.5 = getelementptr inbounds nuw [4 x i8], ptr %row35.5, i64 %g35.5
  store i32 0, ptr %p35.5, align 4
  %row35.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g35.6
  %p35.6 = getelementptr inbounds nuw [4 x i8], ptr %row35.6, i64 %g35.6
  store i32 0, ptr %p35.6, align 4
  %row35.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g35.7
  %p35.7 = getelementptr inbounds nuw [4 x i8], ptr %row35.7, i64 %g35.7
  store i32 0, ptr %p35.7, align 4
  %i.36 = add nuw nsw i64 %i, 288
  %g36.1 = or disjoint i64 %i.36, 1
  %g36.2 = or disjoint i64 %i.36, 2
  %g36.3 = or disjoint i64 %i.36, 3
  %g36.4 = or disjoint i64 %i.36, 4
  %g36.5 = or disjoint i64 %i.36, 5
  %g36.6 = or disjoint i64 %i.36, 6
  %g36.7 = or disjoint i64 %i.36, 7
  %row36.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.36
  %p36.0 = getelementptr inbounds nuw [4 x i8], ptr %row36.0, i64 %i.36
  store i32 0, ptr %p36.0, align 4
  %row36.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g36.1
  %p36.1 = getelementptr inbounds nuw [4 x i8], ptr %row36.1, i64 %g36.1
  store i32 0, ptr %p36.1, align 4
  %row36.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g36.2
  %p36.2 = getelementptr inbounds nuw [4 x i8], ptr %row36.2, i64 %g36.2
  store i32 0, ptr %p36.2, align 4
  %row36.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g36.3
  %p36.3 = getelementptr inbounds nuw [4 x i8], ptr %row36.3, i64 %g36.3
  store i32 0, ptr %p36.3, align 4
  %row36.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g36.4
  %p36.4 = getelementptr inbounds nuw [4 x i8], ptr %row36.4, i64 %g36.4
  store i32 0, ptr %p36.4, align 4
  %row36.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g36.5
  %p36.5 = getelementptr inbounds nuw [4 x i8], ptr %row36.5, i64 %g36.5
  store i32 0, ptr %p36.5, align 4
  %row36.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g36.6
  %p36.6 = getelementptr inbounds nuw [4 x i8], ptr %row36.6, i64 %g36.6
  store i32 0, ptr %p36.6, align 4
  %row36.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g36.7
  %p36.7 = getelementptr inbounds nuw [4 x i8], ptr %row36.7, i64 %g36.7
  store i32 0, ptr %p36.7, align 4
  %i.37 = add nuw nsw i64 %i, 296
  %g37.1 = or disjoint i64 %i.37, 1
  %g37.2 = or disjoint i64 %i.37, 2
  %g37.3 = or disjoint i64 %i.37, 3
  %g37.4 = or disjoint i64 %i.37, 4
  %g37.5 = or disjoint i64 %i.37, 5
  %g37.6 = or disjoint i64 %i.37, 6
  %g37.7 = or disjoint i64 %i.37, 7
  %row37.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.37
  %p37.0 = getelementptr inbounds nuw [4 x i8], ptr %row37.0, i64 %i.37
  store i32 0, ptr %p37.0, align 4
  %row37.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g37.1
  %p37.1 = getelementptr inbounds nuw [4 x i8], ptr %row37.1, i64 %g37.1
  store i32 0, ptr %p37.1, align 4
  %row37.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g37.2
  %p37.2 = getelementptr inbounds nuw [4 x i8], ptr %row37.2, i64 %g37.2
  store i32 0, ptr %p37.2, align 4
  %row37.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g37.3
  %p37.3 = getelementptr inbounds nuw [4 x i8], ptr %row37.3, i64 %g37.3
  store i32 0, ptr %p37.3, align 4
  %row37.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g37.4
  %p37.4 = getelementptr inbounds nuw [4 x i8], ptr %row37.4, i64 %g37.4
  store i32 0, ptr %p37.4, align 4
  %row37.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g37.5
  %p37.5 = getelementptr inbounds nuw [4 x i8], ptr %row37.5, i64 %g37.5
  store i32 0, ptr %p37.5, align 4
  %row37.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g37.6
  %p37.6 = getelementptr inbounds nuw [4 x i8], ptr %row37.6, i64 %g37.6
  store i32 0, ptr %p37.6, align 4
  %row37.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g37.7
  %p37.7 = getelementptr inbounds nuw [4 x i8], ptr %row37.7, i64 %g37.7
  store i32 0, ptr %p37.7, align 4
  %i.38 = add nuw nsw i64 %i, 304
  %g38.1 = or disjoint i64 %i.38, 1
  %g38.2 = or disjoint i64 %i.38, 2
  %g38.3 = or disjoint i64 %i.38, 3
  %g38.4 = or disjoint i64 %i.38, 4
  %g38.5 = or disjoint i64 %i.38, 5
  %g38.6 = or disjoint i64 %i.38, 6
  %g38.7 = or disjoint i64 %i.38, 7
  %row38.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.38
  %p38.0 = getelementptr inbounds nuw [4 x i8], ptr %row38.0, i64 %i.38
  store i32 0, ptr %p38.0, align 4
  %row38.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g38.1
  %p38.1 = getelementptr inbounds nuw [4 x i8], ptr %row38.1, i64 %g38.1
  store i32 0, ptr %p38.1, align 4
  %row38.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g38.2
  %p38.2 = getelementptr inbounds nuw [4 x i8], ptr %row38.2, i64 %g38.2
  store i32 0, ptr %p38.2, align 4
  %row38.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g38.3
  %p38.3 = getelementptr inbounds nuw [4 x i8], ptr %row38.3, i64 %g38.3
  store i32 0, ptr %p38.3, align 4
  %row38.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g38.4
  %p38.4 = getelementptr inbounds nuw [4 x i8], ptr %row38.4, i64 %g38.4
  store i32 0, ptr %p38.4, align 4
  %row38.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g38.5
  %p38.5 = getelementptr inbounds nuw [4 x i8], ptr %row38.5, i64 %g38.5
  store i32 0, ptr %p38.5, align 4
  %row38.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g38.6
  %p38.6 = getelementptr inbounds nuw [4 x i8], ptr %row38.6, i64 %g38.6
  store i32 0, ptr %p38.6, align 4
  %row38.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g38.7
  %p38.7 = getelementptr inbounds nuw [4 x i8], ptr %row38.7, i64 %g38.7
  store i32 0, ptr %p38.7, align 4
  %i.39 = add nuw nsw i64 %i, 312
  %g39.1 = or disjoint i64 %i.39, 1
  %g39.2 = or disjoint i64 %i.39, 2
  %g39.3 = or disjoint i64 %i.39, 3
  %g39.4 = or disjoint i64 %i.39, 4
  %g39.5 = or disjoint i64 %i.39, 5
  %g39.6 = or disjoint i64 %i.39, 6
  %g39.7 = or disjoint i64 %i.39, 7
  %row39.0 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %i.39
  %p39.0 = getelementptr inbounds nuw [4 x i8], ptr %row39.0, i64 %i.39
  store i32 0, ptr %p39.0, align 4
  %row39.1 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g39.1
  %p39.1 = getelementptr inbounds nuw [4 x i8], ptr %row39.1, i64 %g39.1
  store i32 0, ptr %p39.1, align 4
  %row39.2 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g39.2
  %p39.2 = getelementptr inbounds nuw [4 x i8], ptr %row39.2, i64 %g39.2
  store i32 0, ptr %p39.2, align 4
  %row39.3 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g39.3
  %p39.3 = getelementptr inbounds nuw [4 x i8], ptr %row39.3, i64 %g39.3
  store i32 0, ptr %p39.3, align 4
  %row39.4 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g39.4
  %p39.4 = getelementptr inbounds nuw [4 x i8], ptr %row39.4, i64 %g39.4
  store i32 0, ptr %p39.4, align 4
  %row39.5 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g39.5
  %p39.5 = getelementptr inbounds nuw [4 x i8], ptr %row39.5, i64 %g39.5
  store i32 0, ptr %p39.5, align 4
  %row39.6 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g39.6
  %p39.6 = getelementptr inbounds nuw [4 x i8], ptr %row39.6, i64 %g39.6
  store i32 0, ptr %p39.6, align 4
  %row39.7 = getelementptr inbounds nuw [20000 x i8], ptr @arr, i64 %g39.7
  %p39.7 = getelementptr inbounds nuw [4 x i8], ptr %row39.7, i64 %g39.7
  store i32 0, ptr %p39.7, align 4
  %i.next = add nuw nsw i64 %i, 320
  %done = icmp eq i64 %i.next, 4992
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
