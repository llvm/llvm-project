; RUN: llc -mtriple=armv7-linux < %s -o - | FileCheck -check-prefix=LINUX %s
; RUN: llc -mtriple=thumbv7-windows-msvc -mattr=+thumb-mode < %s -o - | FileCheck -check-prefix=WINDOWS %s

define double @testExp(double %val, i32 %a) {
; LINUX:    b ldexp{{$}}
; WINDOWS:  b.w ldexp{{$}}
entry:
  %call = tail call double @ldexp(double %val, i32 %a)
  ret double %call
}

declare double @ldexp(double, i32) memory(none)

define double @testExpIntrinsic(double %val, i32 %a) {
; LINUX:    b ldexp{{$}}
; WINDOWS:  b.w ldexp{{$}}
entry:
  %call = tail call double @llvm.ldexp.f64(double %val, i32 %a)
  ret double %call
}

define float @testExpf(float %val, i32 %a) {
; LINUX:    b ldexpf
; WINDOWS:  b.w ldexpf
entry:
  %call = tail call float @ldexpf(float %val, i32 %a)
  ret float %call
}

define float @testExpfIntrinsic(float %val, i32 %a) {
; LINUX:    b ldexpf
; WINDOWS:  bl ldexp{{$}}
entry:
  %call = tail call float @llvm.ldexp.f32(float %val, i32 %a)
  ret float %call
}

declare float @ldexpf(float, i32) memory(none)

define half @testExpf16(half %val, i32 %a) {
; LINUX: bl ldexpf
; WINDOWS: bl ldexp{{$}}
entry:
  %0 = tail call half @llvm.ldexp.f16.i32(half %val, i32 %a)
  ret half %0
}

declare half @llvm.ldexp.f16.i32(half, i32) memory(none)

define fp128 @testExpf128(fp128 %val, i32 %a) {
; LINUX-LABEL: testExpf128:
; LINUX:       @ %bb.0: @ %entry
; LINUX-NEXT:    push {r11, lr}
; LINUX-NEXT:    sub sp, sp, #8
; LINUX-NEXT:    ldr r12, [sp, #16]
; LINUX-NEXT:    str r12, [sp]
; LINUX-NEXT:    bl ldexpl
; LINUX-NEXT:    add sp, sp, #8
; LINUX-NEXT:    pop {r11, pc}
;
; WINDOWS-LABEL: testExpf128:
; WINDOWS:       @ %bb.0: @ %entry
; WINDOWS-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11, lr}
; WINDOWS-NEXT:    .seh_save_regs_w {r4-r11, lr}
; WINDOWS-NEXT:    sub sp, #68
; WINDOWS-NEXT:    .seh_stackalloc 68
; WINDOWS-NEXT:    .seh_endprologue
; WINDOWS-NEXT:    movw r8, #0
; WINDOWS-NEXT:    movs r5, #0
; WINDOWS-NEXT:    movt r8, #32766
; WINDOWS-NEXT:    strd r5, r5, [sp]
; WINDOWS-NEXT:    strd r5, r8, [sp, #8]
; WINDOWS-NEXT:    mov r4, r0
; WINDOWS-NEXT:    str r0, [sp, #28] @ 4-byte Spill
; WINDOWS-NEXT:    mov r7, r1
; WINDOWS-NEXT:    str r1, [sp, #32] @ 4-byte Spill
; WINDOWS-NEXT:    mov r6, r2
; WINDOWS-NEXT:    str r2, [sp, #24] @ 4-byte Spill
; WINDOWS-NEXT:    mov r9, r3
; WINDOWS-NEXT:    str r3, [sp, #20] @ 4-byte Spill
; WINDOWS-NEXT:    bl __multf3
; WINDOWS-NEXT:    str r0, [sp, #64] @ 4-byte Spill
; WINDOWS-NEXT:    strd r2, r1, [sp, #56] @ 8-byte Folded Spill
; WINDOWS-NEXT:    str r3, [sp, #44] @ 4-byte Spill
; WINDOWS-NEXT:    strd r5, r5, [sp]
; WINDOWS-NEXT:    strd r5, r8, [sp, #8]
; WINDOWS-NEXT:    bl __multf3
; WINDOWS-NEXT:    strd r1, r0, [sp, #48] @ 8-byte Folded Spill
; WINDOWS-NEXT:    mov.w r10, #7471104
; WINDOWS-NEXT:    strd r3, r2, [sp, #36] @ 8-byte Folded Spill
; WINDOWS-NEXT:    mov r0, r4
; WINDOWS-NEXT:    mov r1, r7
; WINDOWS-NEXT:    mov r2, r6
; WINDOWS-NEXT:    mov r3, r9
; WINDOWS-NEXT:    strd r5, r5, [sp]
; WINDOWS-NEXT:    strd r5, r10, [sp, #8]
; WINDOWS-NEXT:    bl __multf3
; WINDOWS-NEXT:    mov r11, r0
; WINDOWS-NEXT:    mov r4, r1
; WINDOWS-NEXT:    mov r8, r2
; WINDOWS-NEXT:    mov r9, r3
; WINDOWS-NEXT:    strd r5, r5, [sp]
; WINDOWS-NEXT:    strd r5, r10, [sp, #8]
; WINDOWS-NEXT:    bl __multf3
; WINDOWS-NEXT:    ldr.w r10, [sp, #104]
; WINDOWS-NEXT:    movw r6, #16616
; WINDOWS-NEXT:    movw r7, #32885
; WINDOWS-NEXT:    movt r6, #65535
; WINDOWS-NEXT:    cmp r10, r6
; WINDOWS-NEXT:    movt r7, #65535
; WINDOWS-NEXT:    str r5, [sp, #8]
; WINDOWS-NEXT:    movw r12, #32766
; WINDOWS-NEXT:    strd r5, r5, [sp]
; WINDOWS-NEXT:    it gt
; WINDOWS-NEXT:    movgt r6, r10
; WINDOWS-NEXT:    cmp r10, r7
; WINDOWS-NEXT:    movw r7, #16269
; WINDOWS-NEXT:    add r7, r10
; WINDOWS-NEXT:    itttt hs
; WINDOWS-NEXT:    movhs r1, r4
; WINDOWS-NEXT:    movhs r0, r11
; WINDOWS-NEXT:    movhs r2, r8
; WINDOWS-NEXT:    movhs r3, r9
; WINDOWS-NEXT:    movw r4, #32538
; WINDOWS-NEXT:    it lo
; WINDOWS-NEXT:    addlo r7, r6, r4
; WINDOWS-NEXT:    movw r6, #49154
; WINDOWS-NEXT:    movw lr, #16383
; WINDOWS-NEXT:    movt r6, #65535
; WINDOWS-NEXT:    cmp r10, r6
; WINDOWS-NEXT:    ldr r6, [sp, #20] @ 4-byte Reload
; WINDOWS-NEXT:    it ge
; WINDOWS-NEXT:    movge r3, r6
; WINDOWS-NEXT:    ldr r6, [sp, #24] @ 4-byte Reload
; WINDOWS-NEXT:    it ge
; WINDOWS-NEXT:    movge r2, r6
; WINDOWS-NEXT:    ldr r6, [sp, #28] @ 4-byte Reload
; WINDOWS-NEXT:    it ge
; WINDOWS-NEXT:    movge r0, r6
; WINDOWS-NEXT:    ldr r6, [sp, #32] @ 4-byte Reload
; WINDOWS-NEXT:    itt ge
; WINDOWS-NEXT:    movge r1, r6
; WINDOWS-NEXT:    movge r7, r10
; WINDOWS-NEXT:    movw r6, #49149
; WINDOWS-NEXT:    cmp r10, r6
; WINDOWS-NEXT:    it lt
; WINDOWS-NEXT:    movlt r6, r10
; WINDOWS-NEXT:    ldr.w r8, [sp, #36] @ 4-byte Reload
; WINDOWS-NEXT:    cmp r10, r12
; WINDOWS-NEXT:    ldr r5, [sp, #44] @ 4-byte Reload
; WINDOWS-NEXT:    it ls
; WINDOWS-NEXT:    movls r8, r5
; WINDOWS-NEXT:    ldr.w r9, [sp, #40] @ 4-byte Reload
; WINDOWS-NEXT:    ldr r5, [sp, #56] @ 4-byte Reload
; WINDOWS-NEXT:    it ls
; WINDOWS-NEXT:    movls r9, r5
; WINDOWS-NEXT:    ldr.w r11, [sp, #48] @ 4-byte Reload
; WINDOWS-NEXT:    ldr r4, [sp, #60] @ 4-byte Reload
; WINDOWS-NEXT:    it ls
; WINDOWS-NEXT:    movls r11, r4
; WINDOWS-NEXT:    ldr r4, [sp, #64] @ 4-byte Reload
; WINDOWS-NEXT:    ldr r5, [sp, #52] @ 4-byte Reload
; WINDOWS-NEXT:    it ls
; WINDOWS-NEXT:    movls r5, r4
; WINDOWS-NEXT:    sub.w r4, r10, lr
; WINDOWS-NEXT:    it hi
; WINDOWS-NEXT:    subhi.w r4, r6, r12
; WINDOWS-NEXT:    cmp.w r10, #16384
; WINDOWS-NEXT:    it lt
; WINDOWS-NEXT:    movlt r4, r7
; WINDOWS-NEXT:    add.w r7, r4, lr
; WINDOWS-NEXT:    lsl.w r7, r7, #16
; WINDOWS-NEXT:    str r7, [sp, #12]
; WINDOWS-NEXT:    itttt ge
; WINDOWS-NEXT:    movge r0, r5
; WINDOWS-NEXT:    movge r1, r11
; WINDOWS-NEXT:    movge r2, r9
; WINDOWS-NEXT:    movge r3, r8
; WINDOWS-NEXT:    bl __multf3
; WINDOWS-NEXT:    .seh_startepilogue
; WINDOWS-NEXT:    add sp, #68
; WINDOWS-NEXT:    .seh_stackalloc 68
; WINDOWS-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11, pc}
; WINDOWS-NEXT:    .seh_save_regs_w {r4-r11, lr}
; WINDOWS-NEXT:    .seh_endepilogue
; WINDOWS-NEXT:    .seh_endproc
entry:
  %0 = tail call fp128 @llvm.ldexp.f128.i32(fp128 %val, i32 %a)
  ret fp128 %0
}
