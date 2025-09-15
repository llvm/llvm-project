; REQUIRES: to-be-fixed
; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s -pipeliner-experimental-cg=true | FileCheck %s

; Multiply and accumulate
; CHECK: mpyi([[REG0:r([0-9]+)]],[[REG1:r([0-9]+)]])
; CHECK-NEXT: [[REG1]] = memw(r{{[0-9]+}}++#4)
; CHECK-NEXT: [[REG0]] = memw(r{{[0-9]+}}++#4)
; CHECK-NEXT: endloop0

define i32 @f0(ptr %a0, ptr %a1, i32 %a2) {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v7, %b1 ]
  %v1 = phi ptr [ %a0, %b0 ], [ %v10, %b1 ]
  %v2 = phi ptr [ %a1, %b0 ], [ %v11, %b1 ]
  %v3 = phi i32 [ 0, %b0 ], [ %v8, %b1 ]
  %v4 = load i32, ptr %v1, align 4
  %v5 = load i32, ptr %v2, align 4
  %v6 = mul nsw i32 %v5, %v4
  %v7 = add nsw i32 %v6, %v0
  %v8 = add nsw i32 %v3, 1
  %v9 = icmp eq i32 %v8, 10000
  %v10 = getelementptr i32, ptr %v1, i32 1
  %v11 = getelementptr i32, ptr %v2, i32 1
  br i1 %v9, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 %v7
}
