; RUN: opt -S -passes=simplifycfg -switch-to-lookup=true -keep-loops=false -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; The switch result values all fit in i8, so on a typical target SimplifyCFG
; narrows the lookup table element type from i32 down to i8. DXIL/DXC does not
; support i8, and the DirectX target transform reports i8 as an illegal type, so
; SimplifyCFG must keep the original i32 element type instead of downcasting the
; table to i8.

; CHECK: @switch.table.test = private unnamed_addr constant [4 x i32] [i32 3, i32 1, i32 2, i32 5]
; CHECK-NOT: @switch.table.test = private unnamed_addr constant [4 x i8]

define i32 @test(i32 %i) {
; CHECK-LABEL: define i32 @test(
; CHECK:         getelementptr inbounds [4 x i32], ptr @switch.table.test
; CHECK:         load i32, ptr
entry:
  switch i32 %i, label %def [
    i32 0, label %bb0
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
  ]

bb0:
  br label %ret

bb1:
  br label %ret

bb2:
  br label %ret

bb3:
  br label %ret

def:
  br label %ret

ret:
  %r = phi i32 [ 3, %bb0 ], [ 1, %bb1 ], [ 2, %bb2 ], [ 5, %bb3 ], [ 0, %def ]
  ret i32 %r
}
