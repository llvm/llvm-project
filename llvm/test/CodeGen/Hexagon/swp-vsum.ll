; RUN: llc -mtriple=hexagon -mcpu=hexagonv5 -enable-pipeliner < %s -pipeliner-experimental-cg=true | FileCheck %s
; RUN: llc -mtriple=hexagon -mcpu=hexagonv60 -enable-pipeliner < %s -pipeliner-experimental-cg=true | FileCheck %s --check-prefix=CHECKV60

; Simple vector total.
; CHECK: loop0(.LBB0_[[LOOP:.]],
; CHECK: .LBB0_[[LOOP]]:
; CHECK: add(r{{[0-9]+}},r{{[0-9]+}})
; CHECK-NEXT: memw(r{{[0-9]+}}++#4)
; CHECK-NEXT: endloop0

; V60 does not pipeline due to latencies.
; CHECKV60: memw(r{{[0-9]+}}++#4)
; CHECKV60: add(r{{[0-9]+}},r{{[0-9]+}})

define i32 @f0(ptr %a0, i32 %a1) {
b0:
  br label %b1

b1:                                               ; preds = %b1, %b0
  %v0 = phi i32 [ 0, %b0 ], [ %v4, %b1 ]
  %v1 = phi ptr [ %a0, %b0 ], [ %v7, %b1 ]
  %v2 = phi i32 [ 0, %b0 ], [ %v5, %b1 ]
  %v3 = load i32, ptr %v1, align 4
  %v4 = add nsw i32 %v3, %v0
  %v5 = add nsw i32 %v2, 1
  %v6 = icmp eq i32 %v5, 10000
  %v7 = getelementptr i32, ptr %v1, i32 1
  br i1 %v6, label %b2, label %b1

b2:                                               ; preds = %b1
  ret i32 %v4
}
