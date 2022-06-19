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

define { i32, i32 } @test_multi_use_add(i32 %base, i32 %offset) {
; CHECK-LABEL: test_multi_use_add:
; CHECK-THUMB: movs [[CONST:r[0-9]+]], #28
; CHECK-THUMB: movt [[CONST]], #1

  %prod = mul i32 %offset, 65564
  %sum = add i32 %base, %prod

  %ptr = inttoptr i32 %sum to i32*
  %loaded = load i32, i32* %ptr

  %ret.tmp = insertvalue { i32, i32 } undef, i32 %sum, 0
  %ret = insertvalue { i32, i32 } %ret.tmp, i32 %loaded, 1

  ret { i32, i32 } %ret
}

define i32 @test_new(i32 %x, i32 %y) {
; CHECK-ARM-LABEL: test_new:
; CHECK-ARM:       @ %bb.0: @ %entry
; CHECK-ARM-NEXT:    movw r2, #48047
; CHECK-ARM-NEXT:    mul r1, r1, r2
; CHECK-ARM-NEXT:    add r0, r0, r1, lsl #1
; CHECK-ARM-NEXT:    bx lr
;
; CHECK-THUMB-LABEL: test_new:
; CHECK-THUMB:       @ %bb.0: @ %entry
; CHECK-THUMB-NEXT:    movw r2, #48047
; CHECK-THUMB-NEXT:    muls r1, r2, r1
; CHECK-THUMB-NEXT:    add.w r0, r0, r1, lsl #1
; CHECK-THUMB-NEXT:    bx lr
entry:
  %mul = mul i32 %y, 96094
  %conv = add i32 %mul, %x
  ret i32 %conv
}

; This test was hitting issues with deleted nodes because ComplexPatternFuncMutatesDAG
; was not defined.
@arr_9 = external dso_local local_unnamed_addr global [15 x [25 x [18 x i8]]], align 1
define void @test_mutateddag(i32 %b, i32 %c, i32 %d, i1 %cc) {
; CHECK-THUMB-LABEL: test_mutateddag:
; CHECK-THUMB:       @ %bb.0: @ %entry
; CHECK-THUMB-NEXT:    .save {r4, lr}
; CHECK-THUMB-NEXT:    push {r4, lr}
; CHECK-THUMB-NEXT:    movw r12, #50608
; CHECK-THUMB-NEXT:    movw r4, #51512
; CHECK-THUMB-NEXT:    movt r12, #17917
; CHECK-THUMB-NEXT:    movt r4, #52
; CHECK-THUMB-NEXT:    mla r12, r1, r4, r12
; CHECK-THUMB-NEXT:    mov.w r4, #450
; CHECK-THUMB-NEXT:    lsls r3, r3, #31
; CHECK-THUMB-NEXT:    mul lr, r0, r4
; CHECK-THUMB-NEXT:    movw r0, #48047
; CHECK-THUMB-NEXT:    muls r0, r1, r0
; CHECK-THUMB-NEXT:    movw r1, :lower16:arr_9
; CHECK-THUMB-NEXT:    movt r1, :upper16:arr_9
; CHECK-THUMB-NEXT:    add.w r0, r2, r0, lsl #1
; CHECK-THUMB-NEXT:    movw r2, #24420
; CHECK-THUMB-NEXT:    movt r2, #19356
; CHECK-THUMB-NEXT:    add.w r0, r0, r0, lsl #3
; CHECK-THUMB-NEXT:    add.w r0, r1, r0, lsl #1
; CHECK-THUMB-NEXT:    movw r1, #60920
; CHECK-THUMB-NEXT:    movt r1, #64028
; CHECK-THUMB-NEXT:    add r2, r0
; CHECK-THUMB-NEXT:    add r1, r0
; CHECK-THUMB-NEXT:    movs r0, #0
; CHECK-THUMB-NEXT:    b .LBB19_2
; CHECK-THUMB-NEXT:  .LBB19_1: @ %for.cond1.for.cond.cleanup_crit_edge
; CHECK-THUMB-NEXT:    @ in Loop: Header=BB19_2 Depth=1
; CHECK-THUMB-NEXT:    add r1, lr
; CHECK-THUMB-NEXT:    add r2, lr
; CHECK-THUMB-NEXT:  .LBB19_2: @ %for.cond
; CHECK-THUMB-NEXT:    @ =>This Loop Header: Depth=1
; CHECK-THUMB-NEXT:    @ Child Loop BB19_3 Depth 2
; CHECK-THUMB-NEXT:    movs r4, #0
; CHECK-THUMB-NEXT:  .LBB19_3: @ %for.cond2.preheader
; CHECK-THUMB-NEXT:    @ Parent Loop BB19_2 Depth=1
; CHECK-THUMB-NEXT:    @ => This Inner Loop Header: Depth=2
; CHECK-THUMB-NEXT:    cmp r3, #0
; CHECK-THUMB-NEXT:    str r0, [r1, r4]
; CHECK-THUMB-NEXT:    bne .LBB19_1
; CHECK-THUMB-NEXT:  @ %bb.4: @ %for.cond2.preheader.2
; CHECK-THUMB-NEXT:    @ in Loop: Header=BB19_3 Depth=2
; CHECK-THUMB-NEXT:    str r0, [r2, r4]
; CHECK-THUMB-NEXT:    add r4, r12
; CHECK-THUMB-NEXT:    b .LBB19_3
entry:
  %0 = add i32 %d, -4
  %1 = mul i32 %c, 864846
  %2 = add i32 %1, 1367306604
  br label %for.cond

for.cond:                                         ; preds = %for.cond1.for.cond.cleanup_crit_edge, %for.cond.preheader
  %indvar = phi i32 [ 0, %entry ], [ %indvar.next, %for.cond1.for.cond.cleanup_crit_edge ]
  %3 = mul i32 %indvar, %b
  %4 = add i32 %3, -2
  br label %for.cond2.preheader

for.cond2.preheader:                              ; preds = %for.cond2.preheader.2, %for.cond
  %indvar24 = phi i32 [ 0, %for.cond ], [ %indvar.next25.3, %for.cond2.preheader.2 ]
  %indvar.next25 = or i32 %indvar24, 1
  %l5 = mul i32 %2, %indvar.next25
  %scevgep.1 = getelementptr [15 x [25 x [18 x i8]]], [15 x [25 x [18 x i8]]]* @arr_9, i32 -217196, i32 %4, i32 %0, i32 %l5
  %l7 = bitcast i8* %scevgep.1 to i32*
  store i32 0, i32* %l7, align 1
  br i1 %cc, label %for.cond1.for.cond.cleanup_crit_edge, label %for.cond2.preheader.2

for.cond2.preheader.2:                            ; preds = %for.cond2.preheader
  %indvar.next25.1 = or i32 %indvar24, 2
  %l8 = mul i32 %2, %indvar.next25.1
  %scevgep.2 = getelementptr [15 x [25 x [18 x i8]]], [15 x [25 x [18 x i8]]]* @arr_9, i32 -217196, i32 %4, i32 %0, i32 %l8
  %l10 = bitcast i8* %scevgep.2 to i32*
  store i32 0, i32* %l10, align 1
  %indvar.next25.3 = add i32 %indvar24, 4
  br label %for.cond2.preheader

for.cond1.for.cond.cleanup_crit_edge:             ; preds = %for.cond2.preheader
  %indvar.next = add i32 %indvar, 1
  br label %for.cond
}
