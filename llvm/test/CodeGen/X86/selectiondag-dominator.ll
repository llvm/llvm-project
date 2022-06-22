; Make sure we don't crash because we have a stale dominator tree.
; PR33266
; REQUIRES: asserts
; RUN: llc -o /dev/null -verify-dom-info %s

target triple = "x86_64-unknown-linux-gnu"

@global = external global [8 x [8 x [4 x i8]]], align 2
@global.1 = external global { i8, [3 x i8] }, align 4

define void @patatino() local_unnamed_addr {
bb:
  br label %bb1

bb1:
  br label %bb2

bb2:
  br i1 icmp ne (ptr getelementptr inbounds ({ i8, [3 x i8] }, ptr @global.1, i64 0, i32 0), ptr getelementptr inbounds ([8 x [8 x [4 x i8]]], ptr @global, i64 0, i64 6, i64 6, i64 2)), label %bb4, label %bb3

bb3:
  br i1 icmp eq (i64 ashr (i64 shl (i64 zext (i32 srem (i32 7, i32 zext (i1 icmp eq (ptr getelementptr inbounds ({ i8, [3 x i8] }, ptr @global.1, i64 0, i32 0), ptr getelementptr inbounds ([8 x [8 x [4 x i8]]], ptr @global, i64 0, i64 6, i64 6, i64 2)) to i32)) to i64), i64 56), i64 56), i64 0), label %bb5, label %bb4

bb4:
  %tmp = phi i64 [ ashr (i64 shl (i64 zext (i32 srem (i32 7, i32 zext (i1 icmp eq (ptr getelementptr inbounds ({ i8, [3 x i8] }, ptr @global.1, i64 0, i32 0), ptr getelementptr inbounds ([8 x [8 x [4 x i8]]], ptr @global, i64 0, i64 6, i64 6, i64 2)) to i32)) to i64), i64 56), i64 56), %bb3 ], [ 7, %bb2 ]
  ret void

bb5:
  ret void
}
