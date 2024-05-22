; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s -check-prefix=SVE
; RUN: opt -passes="print<cost-model>" 2>&1 -disable-output -mtriple aarch64-linux-gnu -mattr=+sve2 < %s | FileCheck %s --check-prefix=SVE2

; SRHADD

define void @srhadd_i8_sext_i16_fixed(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'srhadd_i8_sext_i16_fixed'
; SVE:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = sext <16 x i8> %ld1 to <16 x i16>
; SVE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = sext <16 x i8> %ld2 to <16 x i16>
;
; SVE2-LABEL: 'srhadd_i8_sext_i16_fixed'
; SVE2:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = sext <16 x i8> %ld1 to <16 x i16>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = sext <16 x i8> %ld2 to <16 x i16>
;
  %ld1 = load <16 x i8>, ptr %a
  %ld2 = load <16 x i8>, ptr %b
  %ext1 = sext <16 x i8> %ld1 to <16 x i16>
  %ext2 = sext <16 x i8> %ld2 to <16 x i16>
  %add1 = add nuw nsw <16 x i16> %ext1, shufflevector (<16 x i16> insertelement (<16 x i16> poison, i16 1, i64 0), <16 x i16> poison, <16 x i32> zeroinitializer)
  %add2 = add nuw nsw <16 x i16> %add1, %ext2
  %shr = lshr <16 x i16> %add2, shufflevector (<16 x i16> insertelement (<16 x i16> poison, i16 1, i64 0), <16 x i16> poison, <16 x i32> zeroinitializer)
  %trunc = trunc <16 x i16> %shr to <16 x i8>
  store <16 x i8> %trunc, ptr %a
  ret void
}

define void @srhadd_i8_sext_i16_scalable(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'srhadd_i8_sext_i16_scalable'
; SVE:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = sext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
; SVE-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = sext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
;
; SVE2-LABEL: 'srhadd_i8_sext_i16_scalable'
; SVE2:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = sext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = sext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
;
  %ld1 = load <vscale x 16 x i8>, ptr %a
  %ld2 = load <vscale x 16 x i8>, ptr %b
  %ext1 = sext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
  %ext2 = sext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
  %add1 = add nuw nsw <vscale x 16 x i16> %ext1, shufflevector (<vscale x 16 x i16> insertelement (<vscale x 16 x i16> poison, i16 1, i64 0), <vscale x 16 x i16> poison, <vscale x 16 x i32> zeroinitializer)
  %add2 = add nuw nsw <vscale x 16 x i16> %add1, %ext2
  %shr = lshr <vscale x 16 x i16> %add2, shufflevector (<vscale x 16 x i16> insertelement (<vscale x 16 x i16> poison, i16 1, i64 0), <vscale x 16 x i16> poison, <vscale x 16 x i32> zeroinitializer)
  %trunc = trunc <vscale x 16 x i16> %shr to <vscale x 16 x i8>
  store <vscale x 16 x i8> %trunc, ptr %a
  ret void
}

define void @srhadd_i16_sext_i64_scalable(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'srhadd_i16_sext_i64_scalable'
; SVE:       Cost Model: Found an estimated cost of 6 for instruction: %ext1 = sext <vscale x 8 x i16> %ld1 to <vscale x 8 x i64>
; SVE-NEXT:  Cost Model: Found an estimated cost of 6 for instruction: %ext2 = sext <vscale x 8 x i16> %ld2 to <vscale x 8 x i64>
;
; SVE2-LABEL: 'srhadd_i16_sext_i64_scalable'
; SVE2:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = sext <vscale x 8 x i16> %ld1 to <vscale x 8 x i64>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = sext <vscale x 8 x i16> %ld2 to <vscale x 8 x i64>
;
  %ld1 = load <vscale x 8 x i16>, ptr %a
  %ld2 = load <vscale x 8 x i16>, ptr %b
  %ext1 = sext <vscale x 8 x i16> %ld1 to <vscale x 8 x i64>
  %ext2 = sext <vscale x 8 x i16> %ld2 to <vscale x 8 x i64>
  %add1 = add nuw nsw <vscale x 8 x i64> %ext1, shufflevector (<vscale x 8 x i64> insertelement (<vscale x 8 x i64> poison, i64 1, i64 0), <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer)
  %add2 = add nuw nsw <vscale x 8 x i64> %add1, %ext2
  %shr = lshr <vscale x 8 x i64> %add2, shufflevector (<vscale x 8 x i64> insertelement (<vscale x 8 x i64> poison, i64 1, i64 0), <vscale x 8 x i64> poison, <vscale x 8 x i32> zeroinitializer)
  %trunc = trunc <vscale x 8 x i64> %shr to <vscale x 8 x i16>
  store <vscale x 8 x i16> %trunc, ptr %a
  ret void
}

; URHADD

define void @urhadd_i32_zext_i64_fixed(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'urhadd_i32_zext_i64_fixed'
; SVE:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = zext <4 x i32> %ld1 to <4 x i64>
; SVE-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = zext <4 x i32> %ld2 to <4 x i64>
;
; SVE2-LABEL: 'urhadd_i32_zext_i64_fixed'
; SVE2:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = zext <4 x i32> %ld1 to <4 x i64>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = zext <4 x i32> %ld2 to <4 x i64>
;
  %ld1 = load <4 x i32>, ptr %a
  %ld2 = load <4 x i32>, ptr %b
  %ext1 = zext <4 x i32> %ld1 to <4 x i64>
  %ext2 = zext <4 x i32> %ld2 to <4 x i64>
  %add1 = add nuw nsw <4 x i64> %ext1, shufflevector (<4 x i64> insertelement (<4 x i64> poison, i64 1, i64 0), <4 x i64> poison, <4 x i32> zeroinitializer)
  %add2 = add nuw nsw <4 x i64> %add1, %ext2
  %shr = lshr <4 x i64> %add2, shufflevector (<4 x i64> insertelement (<4 x i64> poison, i64 1, i64 0), <4 x i64> poison, <4 x i32> zeroinitializer)
  %trunc = trunc <4 x i64> %shr to <4 x i32>
  store <4 x i32> %trunc, ptr %a
  ret void
}

define void @urhadd_i8_zext_i64(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'urhadd_i8_zext_i64'
; SVE:       Cost Model: Found an estimated cost of 14 for instruction: %ext1 = zext <vscale x 16 x i8> %ld1 to <vscale x 16 x i64>
; SVE-NEXT:  Cost Model: Found an estimated cost of 14 for instruction: %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i64>
;
; SVE2-LABEL: 'urhadd_i8_zext_i64'
; SVE2:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = zext <vscale x 16 x i8> %ld1 to <vscale x 16 x i64>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i64>
;
  %ld1 = load <vscale x 16 x i8>, ptr %a
  %ld2 = load <vscale x 16 x i8>, ptr %b
  %ext1 = zext <vscale x 16 x i8> %ld1 to <vscale x 16 x i64>
  %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i64>
  %add1 = add nuw nsw <vscale x 16 x i64> %ext1, shufflevector (<vscale x 16 x i64> insertelement (<vscale x 16 x i64> poison, i64 1, i64 0), <vscale x 16 x i64> poison, <vscale x 16 x i32> zeroinitializer)
  %add2 = add nuw nsw <vscale x 16 x i64> %add1, %ext2
  %shr = lshr <vscale x 16 x i64> %add2, shufflevector (<vscale x 16 x i64> insertelement (<vscale x 16 x i64> poison, i64 1, i64 0), <vscale x 16 x i64> poison, <vscale x 16 x i32> zeroinitializer)
  %trunc = trunc <vscale x 16 x i64> %shr to <vscale x 16 x i8>
  store <vscale x 16 x i8> %trunc, ptr %a
  ret void
}

define void @urhadd_i16_zext_i32(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'urhadd_i16_zext_i32'
; SVE:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = zext <vscale x 8 x i16> %ld1 to <vscale x 8 x i32>
; SVE-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = zext <vscale x 8 x i16> %ld2 to <vscale x 8 x i32>
;
; SVE2-LABEL: 'urhadd_i16_zext_i32'
; SVE2:       Cost Model: Found an estimated cost of 0 for instruction: %ext1 = zext <vscale x 8 x i16> %ld1 to <vscale x 8 x i32>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 0 for instruction: %ext2 = zext <vscale x 8 x i16> %ld2 to <vscale x 8 x i32>
;
  %ld1 = load <vscale x 8 x i16>, ptr %a
  %ld2 = load <vscale x 8 x i16>, ptr %b
  %ext1 = zext <vscale x 8 x i16> %ld1 to <vscale x 8 x i32>
  %ext2 = zext <vscale x 8 x i16> %ld2 to <vscale x 8 x i32>
  %add1 = add nuw nsw <vscale x 8 x i32> %ext1, shufflevector (<vscale x 8 x i32> insertelement (<vscale x 8 x i32> poison, i32 1, i64 0), <vscale x 8 x i32> poison, <vscale x 8 x i32> zeroinitializer)
  %add2 = add nuw nsw <vscale x 8 x i32> %add1, %ext2
  %shr = lshr <vscale x 8 x i32> %add2, shufflevector (<vscale x 8 x i32> insertelement (<vscale x 8 x i32> poison, i32 1, i64 0), <vscale x 8 x i32> poison, <vscale x 8 x i32> zeroinitializer)
  %trunc = trunc <vscale x 8 x i32> %shr to <vscale x 8 x i16>
  store <vscale x 8 x i16> %trunc, ptr %a
  ret void
}

; NEGATIVE TESTS

define void @ext_operand_mismatch(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'ext_operand_mismatch'
; SVE:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = sext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
; SVE-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
;
; SVE2-LABEL: 'ext_operand_mismatch'
; SVE2:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = sext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
;
  %ld1 = load <vscale x 16 x i8>, ptr %a
  %ld2 = load <vscale x 16 x i8>, ptr %b
  %ext1 = sext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
  %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
  %add1 = add nuw nsw <vscale x 16 x i16> %ext1, shufflevector (<vscale x 16 x i16> insertelement (<vscale x 16 x i16> poison, i16 1, i64 0), <vscale x 16 x i16> poison, <vscale x 16 x i32> zeroinitializer)
  %add2 = add nuw nsw <vscale x 16 x i16> %add1, %ext2
  %shr = lshr <vscale x 16 x i16> %add2, shufflevector (<vscale x 16 x i16> insertelement (<vscale x 16 x i16> poison, i16 1, i64 0), <vscale x 16 x i16> poison, <vscale x 16 x i32> zeroinitializer)
  %trunc = trunc <vscale x 16 x i16> %shr to <vscale x 16 x i8>
  store <vscale x 16 x i8> %trunc, ptr %a
  ret void
}

define void @add_multiple_uses(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'add_multiple_uses'
; SVE:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = sext <vscale x 8 x i16> %ld1 to <vscale x 8 x i32>
; SVE-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = sext <vscale x 8 x i16> %ld2 to <vscale x 8 x i32>
;
; SVE2-LABEL: 'add_multiple_uses'
; SVE2:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = sext <vscale x 8 x i16> %ld1 to <vscale x 8 x i32>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = sext <vscale x 8 x i16> %ld2 to <vscale x 8 x i32>
;
  %ld1 = load <vscale x 8 x i16>, ptr %a
  %ld2 = load <vscale x 8 x i16>, ptr %b
  %ext1 = sext <vscale x 8 x i16> %ld1 to <vscale x 8 x i32>
  %ext2 = sext <vscale x 8 x i16> %ld2 to <vscale x 8 x i32>
  %add1 = add nuw nsw <vscale x 8 x i32> %ext1, shufflevector (<vscale x 8 x i32> insertelement (<vscale x 8 x i32> poison, i32 1, i64 0), <vscale x 8 x i32> poison, <vscale x 8 x i32> zeroinitializer)
  %add2 = add nuw nsw <vscale x 8 x i32> %add1, %ext2
  %shr = lshr <vscale x 8 x i32> %add2, shufflevector (<vscale x 8 x i32> insertelement (<vscale x 8 x i32> poison, i32 1, i64 0), <vscale x 8 x i32> poison, <vscale x 8 x i32> zeroinitializer)
  %trunc = trunc <vscale x 8 x i32> %shr to <vscale x 8 x i16>
  %add.res = add nuw nsw <vscale x 8 x i32> %add1, %add2
  %res = trunc <vscale x 8 x i32> %add.res to <vscale x 8 x i16>
  store <vscale x 8 x i16> %res, ptr %a
  ret void
}

define void @shift_multiple_uses(ptr %a, ptr %b, ptr %dst) {
; SVE-LABEL: 'shift_multiple_uses'
; SVE:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = zext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
; SVE-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
;
; SVE2-LABEL: 'shift_multiple_uses'
; SVE2:       Cost Model: Found an estimated cost of 2 for instruction: %ext1 = zext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
; SVE2-NEXT:  Cost Model: Found an estimated cost of 2 for instruction: %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
;
  %ld1 = load <vscale x 16 x i8>, ptr %a
  %ld2 = load <vscale x 16 x i8>, ptr %b
  %ext1 = zext <vscale x 16 x i8> %ld1 to <vscale x 16 x i16>
  %ext2 = zext <vscale x 16 x i8> %ld2 to <vscale x 16 x i16>
  %add1 = add nuw nsw <vscale x 16 x i16> %ext1, shufflevector (<vscale x 16 x i16> insertelement (<vscale x 16 x i16> poison, i16 1, i64 0), <vscale x 16 x i16> poison, <vscale x 16 x i32> zeroinitializer)
  %add2 = add nuw nsw <vscale x 16 x i16> %add1, %ext2
  %shr = lshr <vscale x 16 x i16> %add2, shufflevector (<vscale x 16 x i16> insertelement (<vscale x 16 x i16> poison, i16 1, i64 0), <vscale x 16 x i16> poison, <vscale x 16 x i32> zeroinitializer)
  %trunc = trunc <vscale x 16 x i16> %shr to <vscale x 16 x i8>
  %add3 = add nuw nsw <vscale x 16 x i16> %shr, %add2
  %res = trunc <vscale x 16 x i16> %add3 to <vscale x 16 x i8>
  store <vscale x 16 x i8> %res, ptr %a
  ret void
}
