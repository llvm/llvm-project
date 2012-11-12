; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s -check-prefix=A8
; RUN: llc < %s -mtriple=armv7-apple-darwin -mcpu=cortex-a9 | FileCheck %s -check-prefix=A9
; rdar://8576755


define i32 @test1(i32 %X, i32 %Y, i8 %sh) {
; A8: test1:
; A8: add r0, r0, r1, lsl r2

; A9: test1:
; A9: add r0, r0, r1, lsl r2
        %shift.upgrd.1 = zext i8 %sh to i32
        %A = shl i32 %Y, %shift.upgrd.1
        %B = add i32 %X, %A
        ret i32 %B
}

define i32 @test2(i32 %X, i32 %Y, i8 %sh) {
; A8: test2:
; A8: bic r0, r0, r1, asr r2

; A9: test2:
; A9: bic r0, r0, r1, asr r2
        %shift.upgrd.2 = zext i8 %sh to i32
        %A = ashr i32 %Y, %shift.upgrd.2
        %B = xor i32 %A, -1
        %C = and i32 %X, %B
        ret i32 %C
}

define i32 @test3(i32 %base, i32 %base2, i32 %offset) {
entry:
; A8: test3:
; A8: ldr r0, [r0, r2, lsl #2]
; A8: ldr r1, [r1, r2, lsl #2]

; lsl #2 is free
; A9: test3:
; A9: ldr r0, [r0, r2, lsl #2]
; A9: ldr r1, [r1, r2, lsl #2]
        %tmp1 = shl i32 %offset, 2
        %tmp2 = add i32 %base, %tmp1
        %tmp3 = inttoptr i32 %tmp2 to i32*
        %tmp4 = add i32 %base2, %tmp1
        %tmp5 = inttoptr i32 %tmp4 to i32*
        %tmp6 = load i32* %tmp3
        %tmp7 = load i32* %tmp5
        %tmp8 = add i32 %tmp7, %tmp6
        ret i32 %tmp8
}

declare i8* @malloc(...)

define fastcc void @test4(i16 %addr) nounwind {
entry:
; A8: test4:
; A8: ldr [[REG:r[0-9]+]], [r0, r1, lsl #2]
; A8-NOT: ldr [[REG:r[0-9]+]], [r0, r1, lsl #2]!
; A8: str [[REG]], [r0, r1, lsl #2]
; A8-NOT: str [[REG]], [r0]

; A9: test4:
; A9: ldr [[REG:r[0-9]+]], [r0, r1, lsl #2]
; A9-NOT: ldr [[REG:r[0-9]+]], [r0, r1, lsl #2]!
; A9: str [[REG]], [r0, r1, lsl #2]
; A9-NOT: str [[REG]], [r0]
  %0 = tail call i8* (...)* @malloc(i32 undef) nounwind
  %1 = bitcast i8* %0 to i32*
  %2 = sext i16 %addr to i32
  %3 = getelementptr inbounds i32* %1, i32 %2
  %4 = load i32* %3, align 4
  %5 = add nsw i32 %4, 1
  store i32 %5, i32* %3, align 4
  ret void
}
