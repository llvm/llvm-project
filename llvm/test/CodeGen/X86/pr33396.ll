; Make sure we don't crash because we have stale loop infos.
; REQUIRES: asserts
; RUN: llc -o /dev/null -verify-loop-info %s

target triple = "x86_64-unknown-linux-gnu"

@global = external global [2 x i8], align 2
@global.1 = external global [2 x i8], align 2

define void @patatino(i8 %tinky) {
bb:
  br label %bb1

bb1:
  br i1 icmp ne (i8* getelementptr ([2 x i8], [2 x i8]* @global.1, i64 0, i64 1),
                 i8* getelementptr ([2 x i8], [2 x i8]* @global, i64 0, i64 1)), label %bb2, label %bb3

bb2:
  br label %bb3

bb3:
  %tmp = phi i32 [ 60, %bb2 ],
                 [ sdiv (i32 60, i32 zext (i1 icmp eq (i8* getelementptr ([2 x i8], [2 x i8]* @global.1, i64 0, i64 1),
                                           i8* getelementptr ([2 x i8], [2 x i8]* @global, i64 0, i64 1)) to i32)), %bb1 ]
  %tmp4 = icmp slt i8 %tinky, -4
  br label %bb1
}
