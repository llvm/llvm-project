; RUN: llc %s -mtriple=thumbv8m.main     -o - | FileCheck %s --check-prefixes V8M-COMMON,V8M-LE
; RUN: llc %s -mtriple=thumbebv8m.main   -o - | FileCheck %s --check-prefixes V8M-COMMON,V8M-BE
; RUN: llc %s -mtriple=thumbv8.1m.main   -o - | FileCheck %s --check-prefixes V81M-COMMON,V81M-LE
; RUN: llc %s -mtriple=thumbebv8.1m.main -o - | FileCheck %s --check-prefixes V81M-COMMON,V81M-BE

@arr = hidden local_unnamed_addr global [256 x i32] zeroinitializer, align 4

define i32 @access_i16(i16 signext %idx) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_i16:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    sxth r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_i16:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    sxth r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %idxprom = sext i16 %idx to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

define i32 @access_u16(i16 zeroext %idx) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_u16:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    uxth r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_u16:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    uxth r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %idxprom = zext i16 %idx to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

define i32 @access_i8(i8 signext %idx) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_i8:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    sxtb r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_i8:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    sxtb r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %idxprom = sext i8 %idx to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

define i32 @access_u8(i8 zeroext %idx) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_u8:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    uxtb r0, r0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_u8:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    uxtb r0, r0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %idxprom = zext i8 %idx to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

define i32 @access_i1(i1 signext %idx) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_i1:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    and r0, r0, #1
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    rsbs r0, r0, #0
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    and r0, r0, #1
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_i1:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    and r0, r0, #1
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    rsbs r0, r0, #0
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    and r0, r0, #1
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %idxprom = zext i1 %idx to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

define i32 @access_i5(i5 signext %idx) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_i5:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    sbfx r0, r0, #0, #5
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_i5:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    sbfx r0, r0, #0, #5
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %idxprom = sext i5 %idx to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

define i32 @access_u5(i5 zeroext %idx) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_u5:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    movw r1, :lower16:arr
; V8M-COMMON-NEXT:    and r0, r0, #31
; V8M-COMMON-NEXT:    movt r1, :upper16:arr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_u5:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    movw r1, :lower16:arr
; V81M-COMMON-NEXT:    and r0, r0, #31
; V81M-COMMON-NEXT:    movt r1, :upper16:arr
; V81M-COMMON-NEXT:    ldr.w r0, [r1, r0, lsl #2]
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %idxprom = zext i5 %idx to i32
  %arrayidx = getelementptr inbounds [256 x i32], ptr @arr, i32 0, i32 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  ret i32 %0
}

define i32 @access_i33(i33 %arg) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_i33:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-LE-NEXT:        and r0, r1, #1
; V8M-BE-NEXT:        and r0, r0, #1
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    rsbs r0, r0, #0
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_i33:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-LE-NEXT:        and r0, r1, #1
; V81M-BE-NEXT:        and r0, r0, #1
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    rsbs r0, r0, #0
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %shr = ashr i33 %arg, 32
  %conv = trunc nsw i33 %shr to i32
  ret i32 %conv
}

define i32 @access_u33(i33 %arg) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_u33:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-LE-NEXT:        and r0, r1, #1
; V8M-BE-NEXT:        and r0, r0, #1
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_u33:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-LE-NEXT:        and r0, r1, #1
; V81M-BE-NEXT:        and r0, r0, #1
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %shr = lshr i33 %arg, 32
  %conv = trunc nuw nsw i33 %shr to i32
  ret i32 %conv
}

define i32 @access_i65(ptr byval(i65) %0) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_i65:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    sub sp, #16
; V8M-COMMON-NEXT:    stm.w sp, {r0, r1, r2, r3}
; V8M-LE-NEXT:        ldrb.w r0, [sp, #8]
; V8M-LE-NEXT:        and r0, r0, #1
; V8M-LE-NEXT:        rsbs r0, r0, #0
; V8M-BE-NEXT:        movs r1, #0
; V8M-BE-NEXT:        sub.w r0, r1, r0, lsr #24
; V8M-COMMON-NEXT:    add sp, #16
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_i65:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    sub sp, #16
; V81M-COMMON-NEXT:    add sp, #4
; V81M-COMMON-NEXT:    stm.w sp, {r0, r1, r2, r3}
; V81M-LE-NEXT:        ldrb.w r0, [sp, #8]
; V81M-LE-NEXT:        and r0, r0, #1
; V81M-LE-NEXT:        rsbs r0, r0, #0
; V81M-BE-NEXT:        movs r1, #0
; V81M-BE-NEXT:        sub.w r0, r1, r0, lsr #24
; V81M-COMMON-NEXT:    sub sp, #4
; V81M-COMMON-NEXT:    add sp, #16
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %arg = load i65, ptr %0, align 8
  %shr = ashr i65 %arg, 64
  %conv = trunc nsw i65 %shr to i32
  ret i32 %conv
}

define i32 @access_u65(ptr byval(i65) %0) "cmse_nonsecure_entry" {
; V8M-COMMON-LABEL: access_u65:
; V8M-COMMON:       @ %bb.0: @ %entry
; V8M-COMMON-NEXT:    sub sp, #16
; V8M-COMMON-NEXT:    stm.w sp, {r0, r1, r2, r3}
; V8M-LE-NEXT:        ldrb.w r0, [sp, #8]
; V8M-BE-NEXT:        lsrs r0, r0, #24
; V8M-COMMON-NEXT:    add sp, #16
; V8M-COMMON-NEXT:    mov r1, lr
; V8M-COMMON-NEXT:    mov r2, lr
; V8M-COMMON-NEXT:    mov r3, lr
; V8M-COMMON-NEXT:    mov r12, lr
; V8M-COMMON-NEXT:    msr apsr_nzcvq, lr
; V8M-COMMON-NEXT:    bxns lr
;
; V81M-COMMON-LABEL: access_u65:
; V81M-COMMON:       @ %bb.0: @ %entry
; V81M-COMMON-NEXT:    vstr fpcxtns, [sp, #-4]!
; V81M-COMMON-NEXT:    sub sp, #16
; V81M-COMMON-NEXT:    add sp, #4
; V81M-COMMON-NEXT:    stm.w sp, {r0, r1, r2, r3}
; V81M-LE-NEXT:        ldrb.w r0, [sp, #8]
; V81M-BE-NEXT:        lsrs r0, r0, #24
; V81M-COMMON-NEXT:    sub sp, #4
; V81M-COMMON-NEXT:    add sp, #16
; V81M-COMMON-NEXT:    vscclrm {s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, vpr}
; V81M-COMMON-NEXT:    vldr fpcxtns, [sp], #4
; V81M-COMMON-NEXT:    clrm {r1, r2, r3, r12, apsr}
; V81M-COMMON-NEXT:    bxns lr
entry:
  %arg = load i65, ptr %0, align 8
  %shr = lshr i65 %arg, 64
  %conv = trunc nuw nsw i65 %shr to i32
  ret i32 %conv
}
