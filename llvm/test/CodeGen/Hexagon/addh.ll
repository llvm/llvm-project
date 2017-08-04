; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK: r{{[0-9]+}} = add(r{{[0-9]+}}.{{L|l}},r{{[0-9]+}}.{{L|l}})

define i64 @test_cast(i64 %arg0, i16 zeroext %arg1, i16 zeroext %arg2) nounwind readnone {
entry:
  %conv.i = zext i16 %arg1 to i32
  %conv1.i = zext i16 %arg2 to i32
  %sub.i = add nsw i32 %conv.i, %conv1.i
  %sext.i = shl i32 %sub.i, 16
  %cmp.i = icmp slt i32 %sext.i, 65536
  %0 = ashr exact i32 %sext.i, 16
  %conv7.i = select i1 %cmp.i, i32 1, i32 %0
  %cmp8.i = icmp sgt i32 %conv7.i, 4
  %conv7.op.i = add i32 %conv7.i, 65535
  %shl = shl i64 %arg0, 2
  %.mask = and i32 %conv7.op.i, 65535
  %1 = zext i32 %.mask to i64
  %conv = select i1 %cmp8.i, i64 3, i64 %1
  %or = or i64 %conv, %shl
  ret i64 %or
}
