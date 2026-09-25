; RUN: opt -hexagon-vc -S -mtriple=hexagon \
; RUN:   -mcpu=hexagonv68 -mattr=+hvxv68,+hvx-length128b \
; RUN:   -pass-remarks=hexagon-vc -pass-remarks-missed=hexagon-vc \
; RUN:   %s -o /dev/null 2>&1 | FileCheck %s

;; Test that HexagonVectorCombine emits optimization remarks.

;; -- Success: aligned vector memory operations --
; CHECK: remark: {{.*}} aligned vector memory operations

define void @test_align(ptr %p) {
entry:
  %p0 = getelementptr i8, ptr %p, i32 0
  %p1 = getelementptr i8, ptr %p, i32 128
  %p2 = getelementptr i8, ptr %p, i32 256
  %p3 = getelementptr i8, ptr %p, i32 384
  %v0 = load <128 x i8>, ptr %p0, align 1
  %v1 = load <128 x i8>, ptr %p1, align 1
  %v2 = load <128 x i8>, ptr %p2, align 1
  %v3 = load <128 x i8>, ptr %p3, align 1
  store <128 x i8> %v0, ptr %p0, align 1
  store <128 x i8> %v1, ptr %p1, align 1
  store <128 x i8> %v2, ptr %p2, align 1
  store <128 x i8> %v3, ptr %p3, align 1
  ret void
}
