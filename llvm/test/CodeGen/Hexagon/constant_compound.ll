; RUN: llc -march=hexagon < %s 2>&1 | FileCheck %s

; Generating a compound instruction with a constant is not profitable.
; The constant needs to be kept in a register before it is fed to compound
; instruction.
; Before, we are generating
; ra = #65820;
; rb = lsr(rb, #8);
; rc ^= and (rb, ra)
; Now, we are generating
; ra = and (#65820, lsr(ra, #8));
; rb = xor(rb, ra)

; CHECK: and(##65280,lsr(r
; CHECK-NOT : ^= and

define dso_local zeroext i16 @test_compound(i16 zeroext %varA, i16 zeroext %varB) local_unnamed_addr #0 {
entry:
  %tmp = zext i16 %varB to i32
  %tmp1 = and i16 %varA, 255
  %tmp2 = zext i16 %tmp1 to i32
  %.masked.i = and i32 %tmp, 255
  %tmp3 = xor i32 %.masked.i, %tmp2
  %tmp4 = tail call i64 @llvm.hexagon.M4.pmpyw(i32 %tmp3, i32 255) #2
  %tmp5 = trunc i64 %tmp4 to i32
  %tmp6 = and i32 %tmp5, 255
  %tmp7 = tail call i64 @llvm.hexagon.M4.pmpyw(i32 %tmp6, i32 81922) #2
  %tmp8 = trunc i64 %tmp7 to i32
  %tmp9 = xor i32 %tmp8, %tmp
  %tmp10 = lshr i32 %tmp9, 8
  %tmp11 = lshr i16 %varA, 8
  %conv2 = zext i16 %tmp11 to i32
  %tmp12 = and i32 %tmp10, 65280
  %.masked.i7 = and i32 %tmp10, 255
  %tmp13 = xor i32 %.masked.i7, %conv2
  %tmp14 = tail call i64 @llvm.hexagon.M4.pmpyw(i32 %tmp13, i32 255) #2
  %tmp15 = trunc i64 %tmp14 to i32
  %tmp16 = and i32 %tmp15, 255
  %tmp17 = tail call i64 @llvm.hexagon.M4.pmpyw(i32 %tmp16, i32 81922) #2
  %tmp18 = trunc i64 %tmp17 to i32
  %tmp19 = xor i32 %tmp12, %tmp18
  %tmp20 = lshr i32 %tmp19, 8
  %tmp21 = trunc i32 %tmp20 to i16
  ret i16 %tmp21
}

; Function Attrs: nounwind readnone
declare i64 @llvm.hexagon.M4.pmpyw(i32, i32) #1

attributes #0 = { nounwind readnone "target-cpu"="hexagonv65" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }
