; REQUIRES: x86_64-linux
; RUN: llc -pseudo-probe-for-profiling %s -filetype=asm -o - | FileCheck %s

declare i32 @f1()
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #0

define void @foo2() {
bb:
; CHECK: .pseudoprobe	2494702099028631698 1 0 0
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 1, i32 0, i64 -1)
  %tmp = call i32 @f1()
  %tmp1 = icmp eq i32 %tmp, 1
  br i1 %tmp1, label %bb5, label %bb8

bb2:
;; Check the pseudo probe with id 2 only has one copy.
; CHECK-COUNT-1: .pseudoprobe	2494702099028631698 2 0 2
; CHECK-NOT: .pseudoprobe	2494702099028631698 2 0 2
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 2, i32 2, i64 -1)
  %tmp4 = icmp ne i32 %tmp, 1
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 2, i32 2, i64 -1)
  switch i1 %tmp4, label %bb2 [
  i1 0, label %bb5
  i1 1, label %bb8
  ]

bb5:
  %tmp6 = phi i1 [ %tmp1, %bb ], [ false, %bb2 ]
  call void @llvm.pseudoprobe(i64 2494702099028631698, i64 2, i32 2, i64 -1)
  br i1 %tmp6, label %bb8, label %bb7

bb7:
  br label %bb8

bb8:
  ret void
}

!llvm.pseudo_probe_desc = !{!0}

!0 = !{i64 2494702099028631698, i64 281612674956943, !"foo2", null}