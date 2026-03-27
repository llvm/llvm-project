; RUN: llc -mtriple=mipsel -mattr=+dsp < %s

@g1 = common global i64 0, align 8
@g2 = common global i64 0, align 8
@g3 = common global i64 0, align 8

define i64 @test_acreg_copy(i32 %a0, i32 %a1, i32 %a2, i32 %a3) {
entry:
  %0 = load i64, ptr @g1, align 8
  %1 = tail call i64 @llvm.mips.maddu(i64 %0, i32 %a0, i32 %a1)
  %2 = tail call i64 @llvm.mips.maddu(i64 %0, i32 %a2, i32 %a3)
  store i64 %1, ptr @g1, align 8
  store i64 %2, ptr @g2, align 8
  tail call void @foo1()
  store i64 %2, ptr @g3, align 8
  ret i64 %1
}

declare i64 @llvm.mips.maddu(i64, i32, i32)

declare void @foo1()

@g4 = common global <2 x i16> zeroinitializer, align 4
@g5 = common global <2 x i16> zeroinitializer, align 4
@g6 = common global <2 x i16> zeroinitializer, align 4

define { i32 } @test_ccond_spill(i32 %a.coerce, i32 %b.coerce) {
entry:
  %0 = bitcast i32 %a.coerce to <2 x i16>
  %1 = bitcast i32 %b.coerce to <2 x i16>
  %cmp3 = icmp slt <2 x i16> %0, %1
  %sext = sext <2 x i1> %cmp3 to <2 x i16>
  store <2 x i16> %sext, ptr @g4, align 4
  tail call void @foo1()
  %2 = load <2 x i16>, ptr @g5, align 4
  %3 = load <2 x i16>, ptr @g6, align 4
  %or = select <2 x i1> %cmp3, <2 x i16> %2, <2 x i16> %3
  %4 = bitcast <2 x i16> %or to i32
  %.fca.0.insert = insertvalue { i32 } undef, i32 %4, 0
  ret { i32 } %.fca.0.insert
}
