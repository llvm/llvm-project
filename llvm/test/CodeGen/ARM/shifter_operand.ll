; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-ARM
; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a9 | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-ARM
; RUN: llc < %s -mtriple=thumbv7m-none-eabi | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-THUMB
; rdar://8576755


define i32 @test1(i32 %X, i32 %Y, i8 %sh) {
; CHECK-LABEL: test1:
; CHECK-ARM: add r0, r0, r1, lsl r2
; CHECK-THUMB: lsls r1, r2
; CHECK-THUMB: add r0, r1
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = shl i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @test2(i32 %X, i32 %Y, i8 %sh) {
; CHECK-LABEL: test2:
; CHECK-ARM: bic r0, r0, r1, asr r2
; CHECK-THUMB: asrs r1, r2
; CHECK-THUMB: bics r0, r1
        %shift.upgrd.2 = zext i8 %sh to i32
        %A = ashr i32 %Y, %shift.upgrd.2
        %B = xor i32 %A, -1
        %C = and i32 %X, %B
        ret i32 %C
}

define i32 @test3(i32 %base, i32 %base2, i32 %offset) {
entry:
; CHECK-LABEL: test3:
; CHECK: ldr{{(.w)?}} r0, [r0, r2, lsl #2]
; CHECK: ldr{{(.w)?}} r1, [r1, r2, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        %tmp4 = add i32 %base2, %tmp1
        %tmp5 = inttoptr i32 %tmp4 to i32*
        %tmp6 = load i32, i32* %tmp3
        %tmp7 = load i32, i32* %tmp5
        %tmp8 = add i32 %tmp7, %tmp6
        ret i32 %tmp8
}

declare i8* @malloc(...)

define fastcc void @test4(i16 %addr) nounwind {
entry:
; CHECK-LABEL: test4:
; CHECK: ldr{{(.w)?}} [[REG:r[0-9]+]], [r0, r1, lsl #2]
; CHECK-NOT: ldr{{(.w)?}} [[REG:r[0-9]+]], [r0, r1, lsl #2]!
; CHECK: str{{(.w)?}} [[REG]], [r0, r1, lsl #2]
; CHECK-NOT: str{{(.w)?}} [[REG]], [r0]
  %0 = tail call i8* (...) @malloc(i32 undef) nounwind
  %1 = bitcast i8* %0 to i32*
  %2 = sext i16 %addr to i32
  %3 = getelementptr inbounds i32, i32* %1, i32 %2
  %4 = load i32, i32* %3, align 4
  %5 = add nsw i32 %4, 1
  store i32 %5, i32* %3, align 4
  ret void
}

define i32 @test_orr_extract_from_mul_1(i32 %x, i32 %y) {
entry:
; CHECK-LABEL: test_orr_extract_from_mul_1
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-ARM: orr r0, r1, r0
; CHECK-THUMB: muls r1, r2, r1
; CHECk-THUMB: orrs r0, r1
  %mul = mul i32 %y, 63767
  %or = or i32 %mul, %x
  ret i32 %or
}

define i32 @test_orr_extract_from_mul_2(i32 %x, i32 %y) {
; CHECK-LABEL: test_orr_extract_from_mul_2
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: orr{{(.w)?}} r0, r0, r1, lsl #1
entry:
  %mul1 = mul i32 %y, 127534
  %or = or i32 %mul1, %x
  ret i32 %or
}

define i32 @test_orr_extract_from_mul_3(i32 %x, i32 %y) {
; CHECK-LABEL: test_orr_extract_from_mul_3
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: orr{{(.w)?}} r0, r0, r1, lsl #2
entry:
  %mul1 = mul i32 %y, 255068
  %or = or i32 %mul1, %x
  ret i32 %or
}

define i32 @test_orr_extract_from_mul_4(i32 %x, i32 %y) {
; CHECK-LABEL: test_orr_extract_from_mul_4
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: orr{{(.w)?}} r0, r0, r1, lsl #3
entry:
  %mul1 = mul i32 %y, 510136
  %or = or i32 %mul1, %x
  ret i32 %or
}

define i32 @test_orr_extract_from_mul_5(i32 %x, i32 %y) {
; CHECK-LABEL: test_orr_extract_from_mul_5
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: orr{{(.w)?}} r0, r0, r1, lsl #4
entry:
  %mul1 = mul i32 %y, 1020272
  %or = or i32 %mul1, %x
  ret i32 %or
}

define i32 @test_orr_extract_from_mul_6(i32 %x, i32 %y) {
; CHECK-LABEL: test_orr_extract_from_mul_6
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: orr{{(.w)?}} r0, r0, r1, lsl #16
entry:
  %mul = mul i32 %y, -115933184
  %or = or i32 %mul, %x
  ret i32 %or
}

define i32 @test_load_extract_from_mul_1(i8* %x, i32 %y) {
; CHECK-LABEL: test_load_extract_from_mul_1
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: ldrb r0, [r0, r1]
entry:
  %mul = mul i32 %y, 63767
  %arrayidx = getelementptr inbounds i8, i8* %x, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @test_load_extract_from_mul_2(i8* %x, i32 %y) {
; CHECK-LABEL: test_load_extract_from_mul_2
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: ldrb{{(.w)?}} r0, [r0, r1, lsl #1]
entry:
  %mul1 = mul i32 %y, 127534
  %arrayidx = getelementptr inbounds i8, i8* %x, i32 %mul1
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @test_load_extract_from_mul_3(i8* %x, i32 %y) {
; CHECK-LABEL: test_load_extract_from_mul_3
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: ldrb{{(.w)?}} r0, [r0, r1, lsl #2]
entry:
  %mul1 = mul i32 %y, 255068
  %arrayidx = getelementptr inbounds i8, i8* %x, i32 %mul1
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @test_load_extract_from_mul_4(i8* %x, i32 %y) {
; CHECK-LABEL: test_load_extract_from_mul_4
; CHECK: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-THUMB: muls r1, r2, r1
; CHECK: ldrb{{(.w)?}} r0, [r0, r1, lsl #3]
entry:
  %mul1 = mul i32 %y, 510136
  %arrayidx = getelementptr inbounds i8, i8* %x, i32 %mul1
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @test_load_extract_from_mul_5(i8* %x, i32 %y) {
; CHECK-LABEL: test_load_extract_from_mul_5
; CHECK-ARM: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-ARM: ldrb r0, [r0, r1, lsl #4]
; CHECK-THUMB: movw r2, #37232
; CHECK-THUMB: movt r2, #15
; CHECK-THUMB: muls r1, r2, r1
; CHECK-THUMB: ldrb r0, [r0, r1]
entry:
  %mul1 = mul i32 %y, 1020272
  %arrayidx = getelementptr inbounds i8, i8* %x, i32 %mul1
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @test_load_extract_from_mul_6(i8* %x, i32 %y) {
; CHECK-LABEL: test_load_extract_from_mul_6
; CHECK-ARM: movw r2, #63767
; CHECK-ARM: mul r1, r1, r2
; CHECK-ARM: ldrb r0, [r0, r1, lsl #16]
; CHECK-THUMB: movs r2, #0
; CHECK-THUMB: movt r2, #63767
; CHECK-THUMB: muls r1, r2, r1
; CHECK-THUMB: ldrb r0, [r0, r1]
entry:
  %mul = mul i32 %y, -115933184
  %arrayidx = getelementptr inbounds i8, i8* %x, i32 %mul
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}


define void @test_well_formed_dag(i32 %in1, i32 %in2, i32* %addr) {
; CHECK-LABEL: test_well_formed_dag:
; CHECK-ARM: movw [[SMALL_CONST:r[0-9]+]], #675
; CHECK-ARM: mul [[SMALL_PROD:r[0-9]+]], r0, [[SMALL_CONST]]
; CHECK-ARM: add {{r[0-9]+}}, r1, [[SMALL_PROD]], lsl #7

  %mul.small = mul i32 %in1, 675
  store i32 %mul.small, i32* %addr
  %mul.big = mul i32 %in1, 86400
  %add = add i32 %in2, %mul.big
  store i32 %add, i32* %addr
  ret void
}
