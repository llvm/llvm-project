; RUN: llc < %s -mtriple=arm64-eabi -aarch64-neon-syntax=apple | FileCheck %s
;
; <rdar://problem/14486451>

%struct.a = type [256 x i16]
%struct.b = type [256 x i32]
%struct.c = type [256 x i64]

define i16 @load_halfword(ptr %ctx, i32 %xor72) nounwind {
; CHECK-LABEL: load_halfword:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: ldrh w0, [x0, [[REG]], lsl #1]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.a, ptr %ctx, i64 0, i64 %idxprom83
  %result = load i16, ptr %arrayidx86, align 2
  ret i16 %result
}

define i32 @load_word(ptr %ctx, i32 %xor72) nounwind {
; CHECK-LABEL: load_word:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: ldr w0, [x0, [[REG]], lsl #2]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.b, ptr %ctx, i64 0, i64 %idxprom83
  %result = load i32, ptr %arrayidx86, align 4
  ret i32 %result
}

define i64 @load_doubleword(ptr %ctx, i32 %xor72) nounwind {
; CHECK-LABEL: load_doubleword:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: ldr x0, [x0, [[REG]], lsl #3]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.c, ptr %ctx, i64 0, i64 %idxprom83
  %result = load i64, ptr %arrayidx86, align 8
  ret i64 %result
}

define void @store_halfword(ptr %ctx, i32 %xor72, i16 %val) nounwind {
; CHECK-LABEL: store_halfword:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: strh w2, [x0, [[REG]], lsl #1]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.a, ptr %ctx, i64 0, i64 %idxprom83
  store i16 %val, ptr %arrayidx86, align 8
  ret void
}

define void @store_word(ptr %ctx, i32 %xor72, i32 %val) nounwind {
; CHECK-LABEL: store_word:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: str w2, [x0, [[REG]], lsl #2]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.b, ptr %ctx, i64 0, i64 %idxprom83
  store i32 %val, ptr %arrayidx86, align 8
  ret void
}

define void @store_doubleword(ptr %ctx, i32 %xor72, i64 %val) nounwind {
; CHECK-LABEL: store_doubleword:
; CHECK: ubfx [[REG:x[0-9]+]], x1, #9, #8
; CHECK: str x2, [x0, [[REG]], lsl #3]
  %shr81 = lshr i32 %xor72, 9
  %conv82 = zext i32 %shr81 to i64
  %idxprom83 = and i64 %conv82, 255
  %arrayidx86 = getelementptr inbounds %struct.c, ptr %ctx, i64 0, i64 %idxprom83
  store i64 %val, ptr %arrayidx86, align 8
  ret void
}

; Check that we combine a shift into the offset instead of using a narrower load
; when we have a load followed by a trunc

define i32 @load_doubleword_trunc_word(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_word:
; CHECK: ldr x0, [x0, x1, lsl #3]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i32
  ret i32 %trunc
}

define i16 @load_doubleword_trunc_halfword(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_halfword:
; CHECK: ldr x0, [x0, x1, lsl #3]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i16
  ret i16 %trunc
}

define i8 @load_doubleword_trunc_byte(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_byte:
; CHECK: ldr x0, [x0, x1, lsl #3]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i8
  ret i8 %trunc
}

define i16 @load_word_trunc_halfword(ptr %ptr, i64 %off) {
entry:
; CHECK-LABEL: load_word_trunc_halfword:
; CHECK: ldr w0, [x0, x1, lsl #2]
  %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
  %x = load i32, ptr %idx, align 8
  %trunc = trunc i32 %x to i16
  ret i16 %trunc
}

define i8 @load_word_trunc_byte(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_word_trunc_byte:
; CHECK: ldr w0, [x0, x1, lsl #2]
entry:
 %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
 %x = load i32, ptr %idx, align 8
 %trunc = trunc i32 %x to i8
 ret i8 %trunc
}

define i8 @load_halfword_trunc_byte(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_halfword_trunc_byte:
; CHECK: ldrh w0, [x0, x1, lsl #1]
entry:
 %idx = getelementptr inbounds i16, ptr %ptr, i64 %off
 %x = load i16, ptr %idx, align 8
 %trunc = trunc i16 %x to i8
 ret i8 %trunc
}

; Check that we do use a narrower load, and so don't combine the shift, when
; the loaded value is zero-extended.

define i64 @load_doubleword_trunc_word_zext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_word_zext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #3
; CHECK: ldr w0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i32
  %ext = zext i32 %trunc to i64
  ret i64 %ext
}

define i64 @load_doubleword_trunc_halfword_zext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_halfword_zext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #3
; CHECK: ldrh w0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i16
  %ext = zext i16 %trunc to i64
  ret i64 %ext
}

define i64 @load_doubleword_trunc_byte_zext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_byte_zext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #3
; CHECK: ldrb w0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i8
  %ext = zext i8 %trunc to i64
  ret i64 %ext
}

define i64 @load_word_trunc_halfword_zext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_word_trunc_halfword_zext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #2
; CHECK: ldrh w0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
  %x = load i32, ptr %idx, align 8
  %trunc = trunc i32 %x to i16
  %ext = zext i16 %trunc to i64
  ret i64 %ext
}

define i64 @load_word_trunc_byte_zext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_word_trunc_byte_zext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #2
; CHECK: ldrb w0, [x0, [[REG]]]
entry:
 %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
 %x = load i32, ptr %idx, align 8
 %trunc = trunc i32 %x to i8
 %ext = zext i8 %trunc to i64
 ret i64 %ext
}

define i64 @load_halfword_trunc_byte_zext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_halfword_trunc_byte_zext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #1
; CHECK: ldrb w0, [x0, [[REG]]]
entry:
 %idx = getelementptr inbounds i16, ptr %ptr, i64 %off
 %x = load i16, ptr %idx, align 8
 %trunc = trunc i16 %x to i8
 %ext = zext i8 %trunc to i64
 ret i64 %ext
}

; Check that we do use a narrower load, and so don't combine the shift, when
; the loaded value is sign-extended.

define i64 @load_doubleword_trunc_word_sext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_word_sext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #3
; CHECK: ldrsw x0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i32
  %ext = sext i32 %trunc to i64
  ret i64 %ext
}

define i64 @load_doubleword_trunc_halfword_sext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_halfword_sext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #3
; CHECK: ldrsh x0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i16
  %ext = sext i16 %trunc to i64
  ret i64 %ext
}

define i64 @load_doubleword_trunc_byte_sext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_byte_sext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #3
; CHECK: ldrsb x0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i8
  %ext = sext i8 %trunc to i64
  ret i64 %ext
}

define i64 @load_word_trunc_halfword_sext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_word_trunc_halfword_sext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #2
; CHECK: ldrsh x0, [x0, [[REG]]]
entry:
  %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
  %x = load i32, ptr %idx, align 8
  %trunc = trunc i32 %x to i16
  %ext = sext i16 %trunc to i64
  ret i64 %ext
}

define i64 @load_word_trunc_byte_sext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_word_trunc_byte_sext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #2
; CHECK: ldrsb x0, [x0, [[REG]]]
entry:
 %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
 %x = load i32, ptr %idx, align 8
 %trunc = trunc i32 %x to i8
 %ext = sext i8 %trunc to i64
 ret i64 %ext
}

define i64 @load_halfword_trunc_byte_sext(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_halfword_trunc_byte_sext:
; CHECK: lsl [[REG:x[0-9]+]], x1, #1
; CHECK: ldrsb x0, [x0, [[REG]]]
entry:
 %idx = getelementptr inbounds i16, ptr %ptr, i64 %off
 %x = load i16, ptr %idx, align 8
 %trunc = trunc i16 %x to i8
 %ext = sext i8 %trunc to i64
 ret i64 %ext
}

; Check that we don't combine the shift, and so will use a narrower load, when
; the shift is used more than once.

define i32 @load_doubleword_trunc_word_reuse_shift(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_word_reuse_shift:
; CHECK: lsl x[[REG1:[0-9]+]], x1, #3
; CHECK: ldr w[[REG2:[0-9]+]], [x0, x[[REG1]]]
; CHECK: add w0, w[[REG2]], w[[REG1]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i32
  %lsl = shl i64 %off, 3
  %lsl.trunc = trunc i64 %lsl to i32
  %add = add i32 %trunc, %lsl.trunc
  ret i32 %add
}

define i16 @load_doubleword_trunc_halfword_reuse_shift(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_halfword_reuse_shift:
; CHECK: lsl x[[REG1:[0-9]+]], x1, #3
; CHECK: ldrh w[[REG2:[0-9]+]], [x0, x[[REG1]]]
; CHECK: add w0, w[[REG2]], w[[REG1]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i16
  %lsl = shl i64 %off, 3
  %lsl.trunc = trunc i64 %lsl to i16
  %add = add i16 %trunc, %lsl.trunc
  ret i16 %add
}

define i8 @load_doubleword_trunc_byte_reuse_shift(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_doubleword_trunc_byte_reuse_shift:
; CHECK: lsl x[[REG1:[0-9]+]], x1, #3
; CHECK: ldrb w[[REG2:[0-9]+]], [x0, x[[REG1]]]
; CHECK: add w0, w[[REG2]], w[[REG1]]
entry:
  %idx = getelementptr inbounds i64, ptr %ptr, i64 %off
  %x = load i64, ptr %idx, align 8
  %trunc = trunc i64 %x to i8
  %lsl = shl i64 %off, 3
  %lsl.trunc = trunc i64 %lsl to i8
  %add = add i8 %trunc, %lsl.trunc
  ret i8 %add
}

define i16 @load_word_trunc_halfword_reuse_shift(ptr %ptr, i64 %off) {
entry:
; CHECK-LABEL: load_word_trunc_halfword_reuse_shift:
; CHECK: lsl x[[REG1:[0-9]+]], x1, #2
; CHECK: ldrh w[[REG2:[0-9]+]], [x0, x[[REG1]]]
; CHECK: add w0, w[[REG2]], w[[REG1]]
  %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
  %x = load i32, ptr %idx, align 8
  %trunc = trunc i32 %x to i16
  %lsl = shl i64 %off, 2
  %lsl.trunc = trunc i64 %lsl to i16
  %add = add i16 %trunc, %lsl.trunc
  ret i16 %add
}

define i8 @load_word_trunc_byte_reuse_shift(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_word_trunc_byte_reuse_shift:
; CHECK: lsl x[[REG1:[0-9]+]], x1, #2
; CHECK: ldrb w[[REG2:[0-9]+]], [x0, x[[REG1]]]
; CHECK: add w0, w[[REG2]], w[[REG1]]
entry:
  %idx = getelementptr inbounds i32, ptr %ptr, i64 %off
  %x = load i32, ptr %idx, align 8
  %trunc = trunc i32 %x to i8
  %lsl = shl i64 %off, 2
  %lsl.trunc = trunc i64 %lsl to i8
  %add = add i8 %trunc, %lsl.trunc
  ret i8 %add
}

define i8 @load_halfword_trunc_byte_reuse_shift(ptr %ptr, i64 %off) {
; CHECK-LABEL: load_halfword_trunc_byte_reuse_shift:
; CHECK: lsl x[[REG1:[0-9]+]], x1, #1
; CHECK: ldrb w[[REG2:[0-9]+]], [x0, x[[REG1]]]
; CHECK: add w0, w[[REG2]], w[[REG1]]
entry:
  %idx = getelementptr inbounds i16, ptr %ptr, i64 %off
  %x = load i16, ptr %idx, align 8
  %trunc = trunc i16 %x to i8
  %lsl = shl i64 %off, 1
  %lsl.trunc = trunc i64 %lsl to i8
  %add = add i8 %trunc, %lsl.trunc
  ret i8 %add
}
