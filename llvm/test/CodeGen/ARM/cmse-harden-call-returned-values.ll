; RUN: llc %s -mtriple=thumbv8m.main     -o - | FileCheck %s --check-prefixes V8M-COMMON,V8M-LE
; RUN: llc %s -mtriple=thumbebv8m.main   -o - | FileCheck %s --check-prefixes V8M-COMMON,V8M-BE
; RUN: llc %s -mtriple=thumbv8.1m.main   -o - | FileCheck %s --check-prefixes V81M-COMMON,V81M-LE
; RUN: llc %s -mtriple=thumbebv8.1m.main -o - | FileCheck %s --check-prefixes V81M-COMMON,V81M-BE

@get_idx = hidden local_unnamed_addr global ptr null, align 4
@arr = hidden local_unnamed_addr global [256 x i32] zeroinitializer, align 4

define i32 @access_i16() {
; V8M-COMMON-LABEL: access_i16:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V8M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V8M-COMMON-NEXT:    ldr r0, [r0]
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    sxth r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_i16:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V81M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V81M-COMMON-NEXT:    ldr r0, [r0]
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    sxth r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %0 = load ptr, ptr @get_idx, align 4
  %call = tail call signext i16 %0() "cmse_nonsecure_call"
  %idxprom = sext i16 %call to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

define i32 @access_u16() {
; V8M-COMMON-LABEL: access_u16:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V8M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V8M-COMMON-NEXT:    ldr r0, [r0]
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    uxth r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_u16:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V81M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V81M-COMMON-NEXT:    ldr r0, [r0]
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    uxth r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %0 = load ptr, ptr @get_idx, align 4
  %call = tail call zeroext i16 %0() "cmse_nonsecure_call"
  %idxprom = zext i16 %call to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

define i32 @access_i8() {
; V8M-COMMON-LABEL: access_i8:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V8M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V8M-COMMON-NEXT:    ldr r0, [r0]
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    sxtb r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_i8:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V81M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V81M-COMMON-NEXT:    ldr r0, [r0]
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    sxtb r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %0 = load ptr, ptr @get_idx, align 4
  %call = tail call signext i8 %0() "cmse_nonsecure_call"
  %idxprom = sext i8 %call to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

define i32 @access_u8() {
; V8M-COMMON-LABEL: access_u8:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V8M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V8M-COMMON-NEXT:    ldr r0, [r0]
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    uxtb r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_u8:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V81M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V81M-COMMON-NEXT:    ldr r0, [r0]
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    uxtb r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %0 = load ptr, ptr @get_idx, align 4
  %call = tail call zeroext i8 %0() "cmse_nonsecure_call"
  %idxprom = zext i8 %call to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

define i32 @access_i1() {
; V8M-COMMON-LABEL: access_i1:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V8M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V8M-COMMON-NEXT:    ldr r0, [r0]
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    and r0, r0, #1
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_i1:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V81M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V81M-COMMON-NEXT:    ldr r0, [r0]
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    and r0, r0, #1
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %0 = load ptr, ptr @get_idx, align 4
  %call = tail call zeroext i1 %0() "cmse_nonsecure_call"
  %idxprom = zext i1 %call to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

define i32 @access_i5() {
; V8M-COMMON-LABEL: access_i5:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V8M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V8M-COMMON-NEXT:    ldr r0, [r0]
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    sbfx r0, r0, #0, #5
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_i5:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V81M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V81M-COMMON-NEXT:    ldr r0, [r0]
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    sbfx r0, r0, #0, #5
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %0 = load ptr, ptr @get_idx, align 4
  %call = tail call signext i5 %0() "cmse_nonsecure_call"
  %idxprom = sext i5 %call to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

define i32 @access_u5() {
; V8M-COMMON-LABEL: access_u5:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V8M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V8M-COMMON-NEXT:    ldr r0, [r0]
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    and r0, r0, #31
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_u5:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    movw r0, :lower16:get_idx
; V81M-COMMON-NEXT:    movt r0, :upper16:get_idx
; V81M-COMMON-NEXT:    ldr r0, [r0]
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    and r0, r0, #31
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %0 = load ptr, ptr @get_idx, align 4
  %call = tail call zeroext i5 %0() "cmse_nonsecure_call"
  %idxprom = zext i5 %call to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %1 = load i32, ptr %arrayidx, align 4
  ret i32 %1
}

define i32 @access_i33(ptr %f) {
; V8M-COMMON-LABEL: access_i33:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-LE-NEXT:        and r0, r1, #1
; V8M-BE-NEXT:        and r0, r0, #1
; V8M-COMMON-NEXT:    rsb.w r0, r0, #0
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_i33:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-LE-NEXT:        and r0, r1, #1
; V81M-BE-NEXT:        and r0, r0, #1
; V81M-COMMON-NEXT:    rsb.w r0, r0, #0
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %call = tail call i33 %f() "cmse_nonsecure_call"
  %shr = ashr i33 %call, 32
  %conv = trunc nsw i33 %shr to i32
  ret i32 %conv
}

define i32 @access_u33(ptr %f) {
; V8M-COMMON-LABEL: access_u33:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    push {r7, lr}
; V8M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-COMMON-NEXT:    bic r0, r0, #1
; V8M-COMMON-NEXT:    sub sp, #136
; V8M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V8M-COMMON-NEXT:    mov r1, r0
; V8M-COMMON-NEXT:    mov r2, r0
; V8M-COMMON-NEXT:    mov r3, r0
; V8M-COMMON-NEXT:    mov r4, r0
; V8M-COMMON-NEXT:    mov r5, r0
; V8M-COMMON-NEXT:    mov r6, r0
; V8M-COMMON-NEXT:    mov r7, r0
; V8M-COMMON-NEXT:    mov r8, r0
; V8M-COMMON-NEXT:    mov r9, r0
; V8M-COMMON-NEXT:    mov r10, r0
; V8M-COMMON-NEXT:    mov r11, r0
; V8M-COMMON-NEXT:    mov r12, r0
; V8M-COMMON-NEXT:    msr apsr_nzcvq, r0
; V8M-COMMON-NEXT:    blxns r0
; V8M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V8M-COMMON-NEXT:    add sp, #136
; V8M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V8M-LE-NEXT:        and r0, r1, #1
; V8M-BE-NEXT:        and r0, r0, #1
; V8M-COMMON-NEXT:    pop {r7, pc}
;
; V81M-COMMON-LABEL: access_u33:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    push {r7, lr}
; V81M-COMMON-NEXT:    push.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-COMMON-NEXT:    bic r0, r0, #1
; V81M-COMMON-NEXT:    sub sp, #136
; V81M-COMMON-NEXT:    vlstm sp, {d0 - d15}
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, apsr}
; V81M-COMMON-NEXT:    blxns r0
; V81M-COMMON-NEXT:    vlldm sp, {d0 - d15}
; V81M-COMMON-NEXT:    add sp, #136
; V81M-COMMON-NEXT:    pop.w {r4, r5, r6, r7, r8, r9, r10, r11}
; V81M-LE-NEXT:        and r0, r1, #1
; V81M-BE-NEXT:        and r0, r0, #1
; V81M-COMMON-NEXT:    pop {r7, pc}
entry:
  %call = tail call i33 %f() "cmse_nonsecure_call"
  %shr = lshr i33 %call, 32
  %conv = trunc nuw nsw i33 %shr to i32
  ret i32 %conv
}
