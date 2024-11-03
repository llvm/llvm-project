; RUN: llc -mtriple=amdgcn-- -verify-machineinstrs -o /dev/null %s
; This testcase produces a situation with unused value numbers in subregister
; liveranges that get distributed by ConnectedVNInfoEqClasses.

define amdgpu_kernel void @hoge(i1 %c0, i1 %c1, i1 %c2, i1 %c3, i1 %c4) {
bb:
  %tmp = tail call i32 @llvm.amdgcn.workitem.id.x()
  br i1 %c0, label %bb2, label %bb23

bb2:
  br i1 %c1, label %bb6, label %bb8

bb6:
  %tmp7 = or i64 undef, undef
  br label %bb8

bb8:
  %tmp9 = phi i64 [ %tmp7, %bb6 ], [ undef, %bb2 ]
  %tmp10 = icmp eq i32 %tmp, 0
  br i1 %tmp10, label %bb11, label %bb23

bb11:
  br i1 %c2, label %bb20, label %bb17

bb17:
  br label %bb20

bb20:
  %tmp21 = phi i64 [ undef, %bb17 ], [ %tmp9, %bb11 ]
  %tmp22 = trunc i64 %tmp21 to i32
  br label %bb23

bb23:
  %tmp24 = phi i32 [ %tmp22, %bb20 ], [ undef, %bb8 ], [ undef, %bb ]
  br label %bb25

bb25:
  %tmp26 = phi i32 [ %tmp24, %bb23 ], [ undef, %bb25 ]
  br i1 %c3, label %bb25, label %bb30

bb30:
  br i1 %c4, label %bb32, label %bb34

bb32:
  %tmp33 = zext i32 %tmp26 to i64
  br label %bb34

bb34:
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
