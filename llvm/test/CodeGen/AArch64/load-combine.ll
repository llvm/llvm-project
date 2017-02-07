; RUN: llc < %s -mtriple=arm64-unknown | FileCheck %s

; i8* p; // p is 1 byte aligned
; (i32) p[0] | ((i32) p[1] << 8) | ((i32) p[2] << 16) | ((i32) p[3] << 24)
define i32 @load_i32_by_i8_unaligned(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_unaligned:
; CHECK: ldr		w0, [x0]
; CHECK-NEXT: ret
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = getelementptr inbounds i8, i8* %tmp, i32 0
  %tmp2 = load i8, i8* %tmp, align 1
  %tmp3 = zext i8 %tmp2 to i32
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i32 1
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i32
  %tmp7 = shl nuw nsw i32 %tmp6, 8
  %tmp8 = or i32 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i32 2
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = shl nuw nsw i32 %tmp11, 16
  %tmp13 = or i32 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i32 3
  %tmp15 = load i8, i8* %tmp14, align 1
  %tmp16 = zext i8 %tmp15 to i32
  %tmp17 = shl nuw nsw i32 %tmp16, 24
  %tmp18 = or i32 %tmp13, %tmp17
  ret i32 %tmp18
}

; i8* p; // p is 4 byte aligned
; (i32) p[0] | ((i32) p[1] << 8) | ((i32) p[2] << 16) | ((i32) p[3] << 24)
define i32 @load_i32_by_i8_aligned(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_aligned:
; CHECK: ldr    w0, [x0]
; CHECK-NEXT: ret
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = getelementptr inbounds i8, i8* %tmp, i32 0
  %tmp2 = load i8, i8* %tmp, align 4
  %tmp3 = zext i8 %tmp2 to i32
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i32 1
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i32
  %tmp7 = shl nuw nsw i32 %tmp6, 8
  %tmp8 = or i32 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i32 2
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = shl nuw nsw i32 %tmp11, 16
  %tmp13 = or i32 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i32 3
  %tmp15 = load i8, i8* %tmp14, align 1
  %tmp16 = zext i8 %tmp15 to i32
  %tmp17 = shl nuw nsw i32 %tmp16, 24
  %tmp18 = or i32 %tmp13, %tmp17
  ret i32 %tmp18
}

; i8* p; // p is 4 byte aligned
; ((i32) p[0] << 24) | ((i32) p[1] << 16) | ((i32) p[2] << 8) | (i32) p[3]
define i32 @load_i32_by_i8_bswap(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_bswap:
; CHECK: ldr		w8, [x0]
; CHECK-NEXT: rev	w0, w8
; CHECK-NEXT: ret
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = load i8, i8* %tmp, align 4
  %tmp2 = zext i8 %tmp1 to i32
  %tmp3 = shl nuw nsw i32 %tmp2, 24
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i32 1
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i32
  %tmp7 = shl nuw nsw i32 %tmp6, 16
  %tmp8 = or i32 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i32 2
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = shl nuw nsw i32 %tmp11, 8
  %tmp13 = or i32 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i32 3
  %tmp15 = load i8, i8* %tmp14, align 1
  %tmp16 = zext i8 %tmp15 to i32
  %tmp17 = or i32 %tmp13, %tmp16
  ret i32 %tmp17
}

; i8* p; // p is 8 byte aligned
; (i64) p[0] | ((i64) p[1] << 8) | ((i64) p[2] << 16) | ((i64) p[3] << 24) | ((i64) p[4] << 32) | ((i64) p[5] << 40) | ((i64) p[6] << 48) | ((i64) p[7] << 56)
define i64 @load_i64_by_i8(i64* %arg) {
; CHECK-LABEL: load_i64_by_i8:
; CHECK: ldr		x0, [x0]
; CHECK-NEXT: ret
  %tmp = bitcast i64* %arg to i8*
  %tmp1 = load i8, i8* %tmp, align 8
  %tmp2 = zext i8 %tmp1 to i64
  %tmp3 = getelementptr inbounds i8, i8* %tmp, i64 1
  %tmp4 = load i8, i8* %tmp3, align 1
  %tmp5 = zext i8 %tmp4 to i64
  %tmp6 = shl nuw nsw i64 %tmp5, 8
  %tmp7 = or i64 %tmp6, %tmp2
  %tmp8 = getelementptr inbounds i8, i8* %tmp, i64 2
  %tmp9 = load i8, i8* %tmp8, align 1
  %tmp10 = zext i8 %tmp9 to i64
  %tmp11 = shl nuw nsw i64 %tmp10, 16
  %tmp12 = or i64 %tmp7, %tmp11
  %tmp13 = getelementptr inbounds i8, i8* %tmp, i64 3
  %tmp14 = load i8, i8* %tmp13, align 1
  %tmp15 = zext i8 %tmp14 to i64
  %tmp16 = shl nuw nsw i64 %tmp15, 24
  %tmp17 = or i64 %tmp12, %tmp16
  %tmp18 = getelementptr inbounds i8, i8* %tmp, i64 4
  %tmp19 = load i8, i8* %tmp18, align 1
  %tmp20 = zext i8 %tmp19 to i64
  %tmp21 = shl nuw nsw i64 %tmp20, 32
  %tmp22 = or i64 %tmp17, %tmp21
  %tmp23 = getelementptr inbounds i8, i8* %tmp, i64 5
  %tmp24 = load i8, i8* %tmp23, align 1
  %tmp25 = zext i8 %tmp24 to i64
  %tmp26 = shl nuw nsw i64 %tmp25, 40
  %tmp27 = or i64 %tmp22, %tmp26
  %tmp28 = getelementptr inbounds i8, i8* %tmp, i64 6
  %tmp29 = load i8, i8* %tmp28, align 1
  %tmp30 = zext i8 %tmp29 to i64
  %tmp31 = shl nuw nsw i64 %tmp30, 48
  %tmp32 = or i64 %tmp27, %tmp31
  %tmp33 = getelementptr inbounds i8, i8* %tmp, i64 7
  %tmp34 = load i8, i8* %tmp33, align 1
  %tmp35 = zext i8 %tmp34 to i64
  %tmp36 = shl nuw i64 %tmp35, 56
  %tmp37 = or i64 %tmp32, %tmp36
  ret i64 %tmp37
}

; i8* p; // p is 8 byte aligned
; ((i64) p[0] << 56) | ((i64) p[1] << 48) | ((i64) p[2] << 40) | ((i64) p[3] << 32) | ((i64) p[4] << 24) | ((i64) p[5] << 16) | ((i64) p[6] << 8) | (i64) p[7]
define i64 @load_i64_by_i8_bswap(i64* %arg) {
; CHECK-LABEL: load_i64_by_i8_bswap:
; CHECK: ldr		x8, [x0]
; CHECK-NEXT: rev	x0, x8
; CHECK-NEXT: ret
  %tmp = bitcast i64* %arg to i8*
  %tmp1 = load i8, i8* %tmp, align 8
  %tmp2 = zext i8 %tmp1 to i64
  %tmp3 = shl nuw i64 %tmp2, 56
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i64 1
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i64
  %tmp7 = shl nuw nsw i64 %tmp6, 48
  %tmp8 = or i64 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i64 2
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i64
  %tmp12 = shl nuw nsw i64 %tmp11, 40
  %tmp13 = or i64 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i64 3
  %tmp15 = load i8, i8* %tmp14, align 1
  %tmp16 = zext i8 %tmp15 to i64
  %tmp17 = shl nuw nsw i64 %tmp16, 32
  %tmp18 = or i64 %tmp13, %tmp17
  %tmp19 = getelementptr inbounds i8, i8* %tmp, i64 4
  %tmp20 = load i8, i8* %tmp19, align 1
  %tmp21 = zext i8 %tmp20 to i64
  %tmp22 = shl nuw nsw i64 %tmp21, 24
  %tmp23 = or i64 %tmp18, %tmp22
  %tmp24 = getelementptr inbounds i8, i8* %tmp, i64 5
  %tmp25 = load i8, i8* %tmp24, align 1
  %tmp26 = zext i8 %tmp25 to i64
  %tmp27 = shl nuw nsw i64 %tmp26, 16
  %tmp28 = or i64 %tmp23, %tmp27
  %tmp29 = getelementptr inbounds i8, i8* %tmp, i64 6
  %tmp30 = load i8, i8* %tmp29, align 1
  %tmp31 = zext i8 %tmp30 to i64
  %tmp32 = shl nuw nsw i64 %tmp31, 8
  %tmp33 = or i64 %tmp28, %tmp32
  %tmp34 = getelementptr inbounds i8, i8* %tmp, i64 7
  %tmp35 = load i8, i8* %tmp34, align 1
  %tmp36 = zext i8 %tmp35 to i64
  %tmp37 = or i64 %tmp33, %tmp36
  ret i64 %tmp37
}

; i8* p; // p[1] is 4 byte aligned
; (i32) p[1] | ((i32) p[2] << 8) | ((i32) p[3] << 16) | ((i32) p[4] << 24)
define i32 @load_i32_by_i8_nonzero_offset(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_nonzero_offset:
; CHECK: ldrb  w8, [x0, #1]
; CHECK-NEXT: ldrb  w9, [x0, #2]
; CHECK-NEXT: ldrb  w10, [x0, #3]
; CHECK-NEXT: ldrb  w11, [x0, #4]
; CHECK-NEXT: bfi w8, w9, #8, #8
; CHECK-NEXT: bfi w8, w10, #16, #8
; CHECK-NEXT: bfi w8, w11, #24, #8
; CHECK-NEXT: mov  w0, w8
; CHECK-NEXT: ret
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = getelementptr inbounds i8, i8* %tmp, i32 1
  %tmp2 = load i8, i8* %tmp1, align 4
  %tmp3 = zext i8 %tmp2 to i32
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i32 2
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i32
  %tmp7 = shl nuw nsw i32 %tmp6, 8
  %tmp8 = or i32 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i32 3
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = shl nuw nsw i32 %tmp11, 16
  %tmp13 = or i32 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i32 4
  %tmp15 = load i8, i8* %tmp14, align 1
  %tmp16 = zext i8 %tmp15 to i32
  %tmp17 = shl nuw nsw i32 %tmp16, 24
  %tmp18 = or i32 %tmp13, %tmp17
  ret i32 %tmp18
}

; i8* p; // p[-4] is 4 byte aligned
; (i32) p[-4] | ((i32) p[-3] << 8) | ((i32) p[-2] << 16) | ((i32) p[-1] << 24)
define i32 @load_i32_by_i8_neg_offset(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_neg_offset:
; CHECK: ldurb w8, [x0, #-4]
; CHECK-NEXT: ldurb w9, [x0, #-3]
; CHECK-NEXT: ldurb w10, [x0, #-2]
; CHECK-NEXT: ldurb w11, [x0, #-1]
; CHECK-NEXT: bfi w8, w9, #8, #8
; CHECK-NEXT: bfi w8, w10, #16, #8
; CHECK-NEXT: bfi w8, w11, #24, #8
; CHECK-NEXT: mov  w0, w8
; CHECK-NEXT: ret
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = getelementptr inbounds i8, i8* %tmp, i32 -4
  %tmp2 = load i8, i8* %tmp1, align 4
  %tmp3 = zext i8 %tmp2 to i32
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i32 -3
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i32
  %tmp7 = shl nuw nsw i32 %tmp6, 8
  %tmp8 = or i32 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i32 -2
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = shl nuw nsw i32 %tmp11, 16
  %tmp13 = or i32 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i32 -1
  %tmp15 = load i8, i8* %tmp14, align 1
  %tmp16 = zext i8 %tmp15 to i32
  %tmp17 = shl nuw nsw i32 %tmp16, 24
  %tmp18 = or i32 %tmp13, %tmp17
  ret i32 %tmp18
}

; i8* p; // p[1] is 4 byte aligned
; (i32) p[4] | ((i32) p[3] << 8) | ((i32) p[2] << 16) | ((i32) p[1] << 24)
define i32 @load_i32_by_i8_nonzero_offset_bswap(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_nonzero_offset_bswap:
; CHECK: ldrb  w8, [x0, #4]
; CHECK-NEXT: ldrb  w9, [x0, #3]
; CHECK-NEXT: ldrb  w10, [x0, #2]
; CHECK-NEXT: ldrb  w11, [x0, #1]
; CHECK-NEXT: bfi w8, w9, #8, #8
; CHECK-NEXT: bfi w8, w10, #16, #8
; CHECK-NEXT: bfi w8, w11, #24, #8
; CHECK-NEXT: mov  w0, w8
; CHECK-NEXT: ret
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = getelementptr inbounds i8, i8* %tmp, i32 4
  %tmp2 = load i8, i8* %tmp1, align 1
  %tmp3 = zext i8 %tmp2 to i32
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i32 3
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i32
  %tmp7 = shl nuw nsw i32 %tmp6, 8
  %tmp8 = or i32 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i32 2
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = shl nuw nsw i32 %tmp11, 16
  %tmp13 = or i32 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i32 1
  %tmp15 = load i8, i8* %tmp14, align 4
  %tmp16 = zext i8 %tmp15 to i32
  %tmp17 = shl nuw nsw i32 %tmp16, 24
  %tmp18 = or i32 %tmp13, %tmp17
  ret i32 %tmp18
}

; i8* p; // p[-4] is 4 byte aligned
; (i32) p[-1] | ((i32) p[-2] << 8) | ((i32) p[-3] << 16) | ((i32) p[-4] << 24)
define i32 @load_i32_by_i8_neg_offset_bswap(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_neg_offset_bswap:
; CHECK: ldurb w8, [x0, #-1]
; CHECK-NEXT: ldurb w9, [x0, #-2]
; CHECK-NEXT: ldurb w10, [x0, #-3]
; CHECK-NEXT: ldurb w11, [x0, #-4]
; CHECK-NEXT: bfi w8, w9, #8, #8
; CHECK-NEXT: bfi w8, w10, #16, #8
; CHECK-NEXT: bfi w8, w11, #24, #8
; CHECK-NEXT: mov  w0, w8
; CHECK-NEXT: ret
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = getelementptr inbounds i8, i8* %tmp, i32 -1
  %tmp2 = load i8, i8* %tmp1, align 1
  %tmp3 = zext i8 %tmp2 to i32
  %tmp4 = getelementptr inbounds i8, i8* %tmp, i32 -2
  %tmp5 = load i8, i8* %tmp4, align 1
  %tmp6 = zext i8 %tmp5 to i32
  %tmp7 = shl nuw nsw i32 %tmp6, 8
  %tmp8 = or i32 %tmp7, %tmp3
  %tmp9 = getelementptr inbounds i8, i8* %tmp, i32 -3
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = shl nuw nsw i32 %tmp11, 16
  %tmp13 = or i32 %tmp8, %tmp12
  %tmp14 = getelementptr inbounds i8, i8* %tmp, i32 -4
  %tmp15 = load i8, i8* %tmp14, align 4
  %tmp16 = zext i8 %tmp15 to i32
  %tmp17 = shl nuw nsw i32 %tmp16, 24
  %tmp18 = or i32 %tmp13, %tmp17
  ret i32 %tmp18
}

declare i16 @llvm.bswap.i16(i16)

; i16* p; // p is 4 byte aligned
; (i32) bswap(p[1]) | (i32) bswap(p[0] << 16)
define i32 @load_i32_by_bswap_i16(i32* %arg) {
; CHECK-LABEL: load_i32_by_bswap_i16:
; CHECK: ldr    w8, [x0]
; CHECK-NEXT: rev w0, w8
; CHECK-NEXT: ret

  %tmp = bitcast i32* %arg to i16*
  %tmp1 = load i16, i16* %tmp, align 4
  %tmp11 = call i16 @llvm.bswap.i16(i16 %tmp1)
  %tmp2 = zext i16 %tmp11 to i32
  %tmp3 = getelementptr inbounds i16, i16* %tmp, i32 1
  %tmp4 = load i16, i16* %tmp3, align 1
  %tmp41 = call i16 @llvm.bswap.i16(i16 %tmp4)
  %tmp5 = zext i16 %tmp41 to i32
  %tmp6 = shl nuw nsw i32 %tmp2, 16
  %tmp7 = or i32 %tmp6, %tmp5
  ret i32 %tmp7
}
