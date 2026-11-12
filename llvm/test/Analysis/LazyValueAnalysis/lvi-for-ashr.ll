; RUN: opt -passes=correlated-propagation -S %s | FileCheck %s
; CHECK-LABEL: @test-ashr
; CHECK: bb_then
; CHECK:  %. = select i1 true, i32 3, i32 2
define i32 @test-ashr(i32 %c) {
chk65:
  %cmp = icmp sgt i32 %c, 65
  br i1 %cmp, label %return, label %chk0

chk0:
  %cmp1 = icmp slt i32 %c, 0
  br i1 %cmp, label %return, label %bb_if

bb_if:
  %ashr.val = ashr exact i32 %c, 2
  %cmp2 = icmp sgt i32 %ashr.val, 15
  br i1 %cmp2, label %bb_then, label %return

bb_then:
  %cmp3 = icmp eq i32 %ashr.val, 16
  %. = select i1 %cmp3, i32 3, i32 2
  br label %return

return:
  %retval = phi i32 [0, %chk65], [1, %chk0], [%., %bb_then], [4, %bb_if]
  ret i32 %retval
}
