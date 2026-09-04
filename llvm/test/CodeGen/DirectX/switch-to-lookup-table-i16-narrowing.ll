; RUN: opt -S -passes=simplifycfg -switch-to-lookup=true -keep-loops=false -mtriple=dxil-pc-shadermodel6.3-library %s | FileCheck %s

; The switch result values all fit in i8. DXIL does not support i8, but when
; native 16-bit types are enabled (the dx.nativelowprec module flag, set by
; -enable-16bit-types), the DirectX target transform reports a minimum lookup
; table element width of i16. So SimplifyCFG narrows the table from i32 to i16
; (but never to i8).

; CHECK: @switch.table.test = private unnamed_addr constant [4 x i16] [i16 3, i16 1, i16 2, i16 5]
; CHECK-NOT: @switch.table.test = private unnamed_addr constant [4 x i8]

define i32 @test(i32 %i) {
; CHECK-LABEL: define i32 @test(
; CHECK:         getelementptr inbounds [4 x i16], ptr @switch.table.test
; CHECK:         load i16, ptr
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

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"dx.nativelowprec", i32 1}
