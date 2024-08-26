; RUN: llc -march=hexagon -rdf-opt=0 < %s -pipeliner-experimental-cg=true -hoist-const-loads=false | FileCheck %s

; Test that we fixup a pipelined loop correctly when the number of
; stages is greater than the compile-time loop trip count. In this
; test, there are two prolog stages, but the loop executes only once.
; In the bug, the final CFG contains two iterations of the loop.

; CHECK-NOT: loop0
; CHECK: r{{[0-9]+}} = mpyi
; CHECK-NOT: r{{[0-9]+}} = mpyi

define i32 @f0(ptr %a0) {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v9, %b1 ]
  %v1 = phi i32 [ 0, %b0 ], [ %v8, %b1 ]
  %v2 = load i32, ptr %a0, align 4
  %v3 = add nsw i32 %v1, 1
  %v4 = srem i32 %v2, 3
  %v5 = icmp ne i32 %v4, 0
  %v6 = sub nsw i32 0, %v2
  %v7 = select i1 %v5, i32 %v6, i32 %v2
  %v8 = mul nsw i32 %v3, %v7
  %v9 = add nsw i32 %v0, 1
  %v10 = icmp eq i32 %v9, 1
  br i1 %v10, label %b2, label %b1

b2:                                               ; preds = %b1
  %v11 = phi i32 [ %v8, %b1 ]
  br label %b3

b3:                                               ; preds = %b3, %b2
  ret i32 %v11
}
