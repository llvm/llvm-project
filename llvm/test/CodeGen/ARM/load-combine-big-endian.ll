; RUN: llc < %s -mtriple=armeb-unknown | FileCheck %s
; RUN: llc < %s -mtriple=armv6eb-unknown | FileCheck %s --check-prefix=CHECK-ARMv6

; i8* p; // p is 4 byte aligned
; ((i32) p[0] << 24) | ((i32) p[1] << 16) | ((i32) p[2] << 8) | (i32) p[3]
define i32 @load_i32_by_i8_big_endian(i32* %arg) {
; CHECK-LABEL: load_i32_by_i8_big_endian:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK-ARMv6-LABEL: load_i32_by_i8_big_endian:
; CHECK-ARMv6: ldr r0, [r0]
; CHECK-ARMv6-NEXT: bx  lr
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

; i8* p; // p is 4 byte aligned
; (i32) p[0] | ((i32) p[1] << 8) | ((i32) p[2] << 16) | ((i32) p[3] << 24)
define i32 @load_i32_by_i8_bswap(i32* %arg) {
; BSWAP is not supported by 32 bit target
; CHECK-LABEL: load_i32_by_i8_bswap:
; CHECK: ldr  r0, [r0]
; CHECK: and
; CHECK-NEXT: and
; CHECK-NEXT: orr
; CHECK-NEXT: orr
; CHECK-NEXT: orr
; CHECK-NEXT: mov pc, lr

; CHECK-ARMv6-LABEL: load_i32_by_i8_bswap:
; CHECK-ARMv6: ldr  r0, [r0]
; CHECK-ARMv6-NEXT: rev  r0, r0
; CHECK-ARMv6-NEXT: bx lr
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
; ((i32) (((i16) p[0] << 8) | (i16) p[1]) << 16) | (i32) (((i16) p[3] << 8) | (i16) p[4])
define i32 @load_i32_by_i16_by_i8_big_endian(i32* %arg) {
; CHECK-LABEL: load_i32_by_i16_by_i8_big_endian:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK-ARMv6-LABEL: load_i32_by_i16_by_i8_big_endian:
; CHECK-ARMv6: ldr r0, [r0]
; CHECK-ARMv6-NEXT: bx  lr
  %tmp = bitcast i32* %arg to i8*
  %tmp1 = load i8, i8* %tmp, align 4
  %tmp2 = zext i8 %tmp1 to i16
  %tmp3 = getelementptr inbounds i8, i8* %tmp, i32 1
  %tmp4 = load i8, i8* %tmp3, align 1
  %tmp5 = zext i8 %tmp4 to i16
  %tmp6 = shl nuw nsw i16 %tmp2, 8
  %tmp7 = or i16 %tmp6, %tmp5
  %tmp8 = getelementptr inbounds i8, i8* %tmp, i32 2
  %tmp9 = load i8, i8* %tmp8, align 1
  %tmp10 = zext i8 %tmp9 to i16
  %tmp11 = getelementptr inbounds i8, i8* %tmp, i32 3
  %tmp12 = load i8, i8* %tmp11, align 1
  %tmp13 = zext i8 %tmp12 to i16
  %tmp14 = shl nuw nsw i16 %tmp10, 8
  %tmp15 = or i16 %tmp14, %tmp13
  %tmp16 = zext i16 %tmp7 to i32
  %tmp17 = zext i16 %tmp15 to i32
  %tmp18 = shl nuw nsw i32 %tmp16, 16
  %tmp19 = or i32 %tmp18, %tmp17
  ret i32 %tmp19
}

; i16* p; // p is 4 byte aligned
; ((i32) p[0] << 16) | (i32) p[1]
define i32 @load_i32_by_i16(i32* %arg) {
; CHECK-LABEL: load_i32_by_i16:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK-ARMv6-LABEL: load_i32_by_i16:
; CHECK-ARMv6: ldr r0, [r0]
; CHECK-ARMv6-NEXT: bx  lr
  %tmp = bitcast i32* %arg to i16*
  %tmp1 = load i16, i16* %tmp, align 4
  %tmp2 = zext i16 %tmp1 to i32
  %tmp3 = getelementptr inbounds i16, i16* %tmp, i32 1
  %tmp4 = load i16, i16* %tmp3, align 1
  %tmp5 = zext i16 %tmp4 to i32
  %tmp6 = shl nuw nsw i32 %tmp2, 16
  %tmp7 = or i32 %tmp6, %tmp5
  ret i32 %tmp7
}

; i16* p_16; // p_16 is 4 byte aligned
; i8* p_8 = (i8*) p_16;
; (i32) (p_16[0] << 16) | ((i32) p[2] << 8) | (i32) p[3]
define i32 @load_i32_by_i16_i8(i32* %arg) {
; CHECK-LABEL: load_i32_by_i16_i8:
; CHECK: ldr r0, [r0]
; CHECK-NEXT: mov pc, lr

; CHECK-ARMv6-LABEL: load_i32_by_i16_i8:
; CHECK-ARMv6: ldr r0, [r0]
; CHECK-ARMv6-NEXT: bx  lr
  %tmp = bitcast i32* %arg to i16*
  %tmp1 = bitcast i32* %arg to i8*
  %tmp2 = load i16, i16* %tmp, align 4
  %tmp3 = zext i16 %tmp2 to i32
  %tmp4 = shl nuw nsw i32 %tmp3, 16
  %tmp5 = getelementptr inbounds i8, i8* %tmp1, i32 2
  %tmp6 = load i8, i8* %tmp5, align 1
  %tmp7 = zext i8 %tmp6 to i32
  %tmp8 = shl nuw nsw i32 %tmp7, 8
  %tmp9 = getelementptr inbounds i8, i8* %tmp1, i32 3
  %tmp10 = load i8, i8* %tmp9, align 1
  %tmp11 = zext i8 %tmp10 to i32
  %tmp12 = or i32 %tmp8, %tmp11
  %tmp13 = or i32 %tmp12, %tmp4
  ret i32 %tmp13
}

; i8* p; // p is 8 byte aligned
; (i64) p[0] | ((i64) p[1] << 8) | ((i64) p[2] << 16) | ((i64) p[3] << 24) | ((i64) p[4] << 32) | ((i64) p[5] << 40) | ((i64) p[6] << 48) | ((i64) p[7] << 56)
define i64 @load_i64_by_i8_bswap(i64* %arg) {
; CHECK-LABEL: load_i64_by_i8_bswap:
; CHECK: ldr{{.*}}r0
; CHECK: ldr{{.*}}r0
; CHECK: and
; CHECK-NEXT: and
; CHECK-NEXT: orr
; CHECK-NEXT: orr
; CHECK-NEXT: and
; CHECK-NEXT: orr
; CHECK-NEXT: and
; CHECK-NEXT: orr
; CHECK-NEXT: orr
; CHECK-NEXT: orr
; CHECK: mov pc, lr

; CHECK-ARMv6-LABEL: load_i64_by_i8_bswap:
; CHECK-ARMv6: ldrd  r2, r3, [r0]
; CHECK-ARMv6: rev r0, r3
; CHECK-ARMv6: rev r1, r2
; CHECK-ARMv6: bx  lr
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
define i64 @load_i64_by_i8(i64* %arg) {
; CHECK-LABEL: load_i64_by_i8:
; CHECK: ldr r2, [r0]
; CHECK: ldr r1, [r0, #4]
; CHECK: mov r0, r2
; CHECK: mov pc, lr

; CHECK-ARMv6-LABEL: load_i64_by_i8:
; CHECK-ARMv6: ldrd  r0, r1, [r0]
; CHECK-ARMv6: bx  lr
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
