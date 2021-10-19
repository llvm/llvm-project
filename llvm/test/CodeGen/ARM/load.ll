; RUN: llc -mtriple=thumbv6m-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-T1
; RUN: llc -mtriple=thumbv7m-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-T2


; Register offset

; CHECK-LABEL: ldrsb_rr
; CHECK:    ldrsb   r0, [r0, r1]
define i32 @ldrsb_rr(i8* %p, i32 %n) {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %n
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrsh_rr
; CHECK-T1: lsls    r1, r1, #1
; CHECK-T1: ldrsh   r0, [r0, r1]
; CHECK-T2: ldrsh.w r0, [r0, r1, lsl #1]
define i32 @ldrsh_rr(i16* %p, i32 %n) {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 %n
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrb_rr
; CHECK:    ldrb r0, [r0, r1]
define i32 @ldrb_rr(i8* %p, i32 %n) {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %n
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrh_rr
; CHECK-T1: lsls    r1, r1, #1
; CHECK-T1: ldrh    r0, [r0, r1]
; CHECK-T2: ldrh.w  r0, [r0, r1, lsl #1]
define i32 @ldrh_rr(i16* %p, i32 %n) {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 %n
  %0 = load i16, i16* %arrayidx, align 2
  %conv = zext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldr_rr
; CHECK-T1: lsls    r1, r1, #2
; CHECK-T1: ldr     r0, [r0, r1]
; CHECK-T2: ldr.w   r0, [r0, r1, lsl #2]
define i32 @ldr_rr(i32* %p, i32 %n) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %n
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0
}

; CHECK-LABEL: strb_rr
; CHECK:    strb    r2, [r0, r1]
define void @strb_rr(i8* %p, i32 %n, i32 %x) {
entry:
  %conv = trunc i32 %x to i8
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 %n
  store i8 %conv, i8* %arrayidx, align 1
  ret void
}

; CHECK-LABEL: strh_rr
; CHECK-T1: lsls    r1, r1, #1
; CHECK-T1: strh    r2, [r0, r1]
; CHECK-T2: strh.w  r2, [r0, r1, lsl #1]
define void @strh_rr(i16* %p, i32 %n, i32 %x) {
entry:
  %conv = trunc i32 %x to i16
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 %n
  store i16 %conv, i16* %arrayidx, align 2
  ret void
}

; CHECK-LABEL: str_rr
; CHECK-T1: lsls    r1, r1, #2
; CHECK-T1: str     r2, [r0, r1]
; CHECK-T2: str.w   r2, [r0, r1, lsl #2]
define void @str_rr(i32* %p, i32 %n, i32 %x) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 %n
  store i32 %x, i32* %arrayidx, align 4
  ret void
}


; Immediate offset of zero

; CHECK-LABEL: ldrsb_ri_zero
; CHECK-T1: movs    r1, #0
; CHECK-T1: ldrsb   r0, [r0, r1]
; CHECK-T2: ldrsb.w r0, [r0]
define i32 @ldrsb_ri_zero(i8* %p) {
entry:
  %0 = load i8, i8* %p, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrsh_ri_zero
; CHECK-T1: movs    r1, #0
; CHECK-T1: ldrsh   r0, [r0, r1]
; CHECK-T2: ldrsh.w r0, [r0]
define i32 @ldrsh_ri_zero(i16* %p) {
entry:
  %0 = load i16, i16* %p, align 2
  %conv = sext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrb_ri_zero
; CHECK:    ldrb    r0, [r0]
define i32 @ldrb_ri_zero(i8* %p) {
entry:
  %0 = load i8, i8* %p, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrh_ri_zero
; CHECK:    ldrh    r0, [r0]
define i32 @ldrh_ri_zero(i16* %p) {
entry:
  %0 = load i16, i16* %p, align 2
  %conv = zext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldr_ri_zero
; CHECK:    ldr     r0, [r0]
define i32 @ldr_ri_zero(i32* %p) {
entry:
  %0 = load i32, i32* %p, align 4
  ret i32 %0
}

; CHECK-LABEL: strb_ri_zero
; CHECK:    strb    r1, [r0]
define void @strb_ri_zero(i8* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i8
  store i8 %conv, i8* %p, align 1
  ret void
}

; CHECK-LABEL: strh_ri_zero
; CHECK:    strh    r1, [r0]
define void @strh_ri_zero(i16* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i16
  store i16 %conv, i16* %p, align 2
  ret void
}

; CHECK-LABEL: str_ri_zero
; CHECK:    str     r1, [r0]
define void @str_ri_zero(i32* %p, i32 %x) {
entry:
  store i32 %x, i32* %p, align 4
  ret void
}


; Maximum Thumb-1 immediate offset

; CHECK-LABEL: ldrsb_ri_t1_max
; CHECK-T1: movs    r1, #31
; CHECK-T1: ldrsb   r0, [r0, r1]
; CHECK-T2: ldrsb.w r0, [r0, #31]
define i32 @ldrsb_ri_t1_max(i8* %p) {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 31
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrsh_ri_t1_max
; CHECK-T1: movs    r1, #62
; CHECK-T1: ldrsh   r0, [r0, r1]
; CHECK-T2: ldrsh.w r0, [r0, #62]
define i32 @ldrsh_ri_t1_max(i16* %p) {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 31
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrb_ri_t1_max
; CHECK:    ldrb    r0, [r0, #31]
define i32 @ldrb_ri_t1_max(i8* %p) {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 31
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrh_ri_t1_max
; CHECK:    ldrh    r0, [r0, #62]
define i32 @ldrh_ri_t1_max(i16* %p) {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 31
  %0 = load i16, i16* %arrayidx, align 2
  %conv = zext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldr_ri_t1_max
; CHECK:    ldr     r0, [r0, #124]
define i32 @ldr_ri_t1_max(i32* %p) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 31
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0
}

; CHECK-LABEL: strb_ri_t1_max
; CHECK:    strb    r1, [r0, #31]
define void @strb_ri_t1_max(i8* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i8
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 31
  store i8 %conv, i8* %arrayidx, align 1
  ret void
}

; CHECK-LABEL: strh_ri_t1_max
; CHECK:    strh    r1, [r0, #62]
define void @strh_ri_t1_max(i16* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i16
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 31
  store i16 %conv, i16* %arrayidx, align 2
  ret void
}

; CHECK-LABEL: str_ri_t1_max
; CHECK:    str     r1, [r0, #124]
define void @str_ri_t1_max(i32* %p, i32 %x) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 31
  store i32 %x, i32* %arrayidx, align 4
  ret void
}


; One past maximum Thumb-1 immediate offset

; CHECK-LABEL: ldrsb_ri_t1_too_big
; CHECK-T1: movs    r1, #32
; CHECK-T1: ldrsb   r0, [r0, r1]
; CHECK-T2: ldrsb.w r0, [r0, #32]
define i32 @ldrsb_ri_t1_too_big(i8* %p) {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 32
  %0 = load i8, i8* %arrayidx, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrsh_ri_t1_too_big
; CHECK-T1: movs    r1, #64
; CHECK-T1: ldrsh   r0, [r0, r1]
; CHECK-T2: ldrsh.w r0, [r0, #64]
define i32 @ldrsh_ri_t1_too_big(i16* %p) {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 32
  %0 = load i16, i16* %arrayidx, align 2
  %conv = sext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrb_ri_t1_too_big
; CHECK-T1: movs    r1, #32
; CHECK-T1: ldrb    r0, [r0, r1]
; CHECK-T2: ldrb.w  r0, [r0, #32]
define i32 @ldrb_ri_t1_too_big(i8* %p) {
entry:
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 32
  %0 = load i8, i8* %arrayidx, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrh_ri_t1_too_big
; CHECK-T1: movs    r1, #64
; CHECK-T1: ldrh    r0, [r0, r1]
; CHECK-T2: ldrh.w  r0, [r0, #64]
define i32 @ldrh_ri_t1_too_big(i16* %p) {
entry:
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 32
  %0 = load i16, i16* %arrayidx, align 2
  %conv = zext i16 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldr_ri_t1_too_big
; CHECK-T1: movs    r1, #128
; CHECK-T1: ldr     r0, [r0, r1]
; CHECK-T2: ldr.w   r0, [r0, #128]
define i32 @ldr_ri_t1_too_big(i32* %p) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 32
  %0 = load i32, i32* %arrayidx, align 4
  ret i32 %0
}

; CHECK-LABEL: strb_ri_t1_too_big
; CHECK-T1: movs    r2, #32
; CHECK-T1: strb    r1, [r0, r2]
; CHECK-T2: strb.w  r1, [r0, #32]
define void @strb_ri_t1_too_big(i8* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i8
  %arrayidx = getelementptr inbounds i8, i8* %p, i32 32
  store i8 %conv, i8* %arrayidx, align 1
  ret void
}

; CHECK-LABEL: strh_ri_t1_too_big
; CHECK-T1: movs    r2, #64
; CHECK-T1: strh    r1, [r0, r2]
; CHECK-T2: strh.w  r1, [r0, #64]
define void @strh_ri_t1_too_big(i16* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i16
  %arrayidx = getelementptr inbounds i16, i16* %p, i32 32
  store i16 %conv, i16* %arrayidx, align 2
  ret void
}

; CHECK-LABEL: str_ri_t1_too_big
; CHECK-T1: movs    r2, #128
; CHECK-T1: str     r1, [r0, r2]
; CHECK-T2: str.w   r1, [r0, #128]
define void @str_ri_t1_too_big(i32* %p, i32 %x) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %p, i32 32
  store i32 %x, i32* %arrayidx, align 4
  ret void
}


; Maximum Thumb-2 immediate offset

; CHECK-LABEL: ldrsb_ri_t2_max
; CHECK-T1: ldr     r1, .LCP
; CHECK-T1: ldrsb   r0, [r0, r1]
; CHECK-T2: ldrsb.w r0, [r0, #4095]
define i32 @ldrsb_ri_t2_max(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  %0 = load i8, i8* %add.ptr, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrsh_ri_t2_max
; CHECK-T1: ldr     r1, .LCP
; CHECK-T1: ldrsh   r0, [r0, r1]
; CHECK-T2: ldrsh.w r0, [r0, #4095]
define i32 @ldrsh_ri_t2_max(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = sext i16 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrb_ri_t2_max
; CHECK-T1: ldr     r1, .LCP
; CHECK-T1: ldrb    r0, [r0, r1]
; CHECK-T2: ldrb.w  r0, [r0, #4095]
define i32 @ldrb_ri_t2_max(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  %0 = load i8, i8* %add.ptr, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrh_ri_t2_max
; CHECK-T1: ldr     r1, .LCP
; CHECK-T1: ldrh    r0, [r0, r1]
; CHECK-T2: ldrh.w  r0, [r0, #4095]
define i32 @ldrh_ri_t2_max(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = zext i16 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldr_ri_t2_max
; CHECK-T1: ldr     r1, .LCP
; CHECK-T1: ldr     r0, [r0, r1]
; CHECK-T2: ldr.w   r0, [r0, #4095]
define i32 @ldr_ri_t2_max(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  %0 = bitcast i8* %add.ptr to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

; CHECK-LABEL: strb_ri_t2_max
; CHECK-T1: ldr     r2, .LCP
; CHECK-T1: strb    r1, [r0, r2]
; CHECK-T2: strb.w  r1, [r0, #4095]
define void @strb_ri_t2_max(i8* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i8
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  store i8 %conv, i8* %add.ptr, align 1
  ret void
}

; CHECK-LABEL: strh_ri_t2_max
; CHECK-T1: ldr     r2, .LCP
; CHECK-T1: strh    r1, [r0, r2]
; CHECK-T2: strh.w  r1, [r0, #4095]
define void @strh_ri_t2_max(i8* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i16
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  %0 = bitcast i8* %add.ptr to i16*
  store i16 %conv, i16* %0, align 2
  ret void
}

; CHECK-LABEL: str_ri_t2_max
; CHECK-T1: ldr     r2, .LCP
; CHECK-T1: str     r1, [r0, r2]
; CHECK-T2: str.w   r1, [r0, #4095]
define void @str_ri_t2_max(i8* %p, i32 %x) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4095
  %0 = bitcast i8* %add.ptr to i32*
  store i32 %x, i32* %0, align 4
  ret void
}


; One past maximum Thumb-2 immediate offset

; CHECK-LABEL: ldrsb_ri_t2_too_big
; CHECK-T1: movs    r1, #1
; CHECK-T1: lsls    r1, r1, #12
; CHECK-T2: mov.w   r1, #4096
; CHECK:    ldrsb   r0, [r0, r1]
define i32 @ldrsb_ri_t2_too_big(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  %0 = load i8, i8* %add.ptr, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrsh_ri_t2_too_big
; CHECK-T1: movs    r1, #1
; CHECK-T1: lsls    r1, r1, #12
; CHECK-T2: mov.w   r1, #4096
; CHECK:    ldrsh   r0, [r0, r1]
define i32 @ldrsh_ri_t2_too_big(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = sext i16 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrb_ri_t2_too_big
; CHECK-T1: movs    r1, #1
; CHECK-T1: lsls    r1, r1, #12
; CHECK-T2: mov.w   r1, #4096
; CHECK:    ldrb    r0, [r0, r1]
define i32 @ldrb_ri_t2_too_big(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  %0 = load i8, i8* %add.ptr, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldrh_ri_t2_too_big
; CHECK-T1: movs    r1, #1
; CHECK-T1: lsls    r1, r1, #12
; CHECK-T2: mov.w   r1, #4096
; CHECK:    ldrh    r0, [r0, r1]
define i32 @ldrh_ri_t2_too_big(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = zext i16 %1 to i32
  ret i32 %conv
}

; CHECK-LABEL: ldr_ri_t2_too_big
; CHECK-T1: movs    r1, #1
; CHECK-T1: lsls    r1, r1, #12
; CHECK-T2: mov.w   r1, #4096
; CHECK:    ldr     r0, [r0, r1]
define i32 @ldr_ri_t2_too_big(i8* %p) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  %0 = bitcast i8* %add.ptr to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

; CHECK-LABEL: strb_ri_t2_too_big
; CHECK-T1: movs    r2, #1
; CHECK-T1: lsls    r2, r2, #12
; CHECK-T2: mov.w   r2, #4096
; CHECK:    strb    r1, [r0, r2]
define void @strb_ri_t2_too_big(i8* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i8
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  store i8 %conv, i8* %add.ptr, align 1
  ret void
}

; CHECK-LABEL: strh_ri_t2_too_big
; CHECK-T1: movs    r2, #1
; CHECK-T1: lsls    r2, r2, #12
; CHECK-T2: mov.w   r2, #4096
; CHECK:    strh    r1, [r0, r2]
define void @strh_ri_t2_too_big(i8* %p, i32 %x) {
entry:
  %conv = trunc i32 %x to i16
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  %0 = bitcast i8* %add.ptr to i16*
  store i16 %conv, i16* %0, align 2
  ret void
}

; CHECK-LABEL: str_ri_t2_too_big
; CHECK-T1: movs    r2, #1
; CHECK-T1: lsls    r2, r2, #12
; CHECK-T2: mov.w   r2, #4096
; CHECK:    str     r1, [r0, r2]
define void @str_ri_t2_too_big(i8* %p, i32 %x) {
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 4096
  %0 = bitcast i8* %add.ptr to i32*
  store i32 %x, i32* %0, align 4
  ret void
}


; Negative offset

define i32 @ldrsb_ri_negative(i8* %p) {
; CHECK-T1-LABEL: ldrsb_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #0
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrsb r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrsb_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrsb r0, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  %0 = load i8, i8* %add.ptr, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

define i32 @ldrsh_ri_negative(i8* %p) {
; CHECK-T1-LABEL: ldrsh_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #0
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrsh r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrsh_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrsh r0, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = sext i16 %1 to i32
  ret i32 %conv
}

define i32 @ldrb_ri_negative(i8* %p) {
; CHECK-T1-LABEL: ldrb_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, r0, #1
; CHECK-T1-NEXT:    ldrb r0, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrb_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrb r0, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  %0 = load i8, i8* %add.ptr, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @ldrh_ri_negative(i8* %p) {
; CHECK-T1-LABEL: ldrh_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, r0, #1
; CHECK-T1-NEXT:    ldrh r0, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrh_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrh r0, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = zext i16 %1 to i32
  ret i32 %conv
}

define i32 @ldr_ri_negative(i8* %p) {
; CHECK-T1-LABEL: ldr_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, r0, #1
; CHECK-T1-NEXT:    ldr r0, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldr_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldr r0, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  %0 = bitcast i8* %add.ptr to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

define void @strb_ri_negative(i8* %p, i32 %x) {
; CHECK-T1-LABEL: strb_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, r0, #1
; CHECK-T1-NEXT:    strb r1, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: strb_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    strb r1, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %conv = trunc i32 %x to i8
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  store i8 %conv, i8* %add.ptr, align 1
  ret void
}

define void @strh_ri_negative(i8* %p, i32 %x) {
; CHECK-T1-LABEL: strh_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, r0, #1
; CHECK-T1-NEXT:    strh r1, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: strh_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    strh r1, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %conv = trunc i32 %x to i16
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  %0 = bitcast i8* %add.ptr to i16*
  store i16 %conv, i16* %0, align 2
  ret void
}

define void @str_ri_negative(i8* %p, i32 %x) {
; CHECK-T1-LABEL: str_ri_negative:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, r0, #1
; CHECK-T1-NEXT:    str r1, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: str_ri_negative:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    str r1, [r0, #-1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -1
  %0 = bitcast i8* %add.ptr to i32*
  store i32 %x, i32* %0, align 4
  ret void
}


; Negative 255 offset

define i32 @ldrsb_ri_negative255(i8* %p) {
; CHECK-T1-LABEL: ldrsb_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #254
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrsb r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrsb_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrsb r0, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  %0 = load i8, i8* %add.ptr, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

define i32 @ldrsh_ri_negative255(i8* %p) {
; CHECK-T1-LABEL: ldrsh_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #254
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrsh r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrsh_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrsh r0, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = sext i16 %1 to i32
  ret i32 %conv
}

define i32 @ldrb_ri_negative255(i8* %p) {
; CHECK-T1-LABEL: ldrb_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, #255
; CHECK-T1-NEXT:    ldrb r0, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrb_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrb r0, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  %0 = load i8, i8* %add.ptr, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @ldrh_ri_negative255(i8* %p) {
; CHECK-T1-LABEL: ldrh_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, #255
; CHECK-T1-NEXT:    ldrh r0, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrh_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldrh r0, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = zext i16 %1 to i32
  ret i32 %conv
}

define i32 @ldr_ri_negative255(i8* %p) {
; CHECK-T1-LABEL: ldr_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, #255
; CHECK-T1-NEXT:    ldr r0, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldr_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    ldr r0, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  %0 = bitcast i8* %add.ptr to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

define void @strb_ri_negative255(i8* %p, i32 %x) {
; CHECK-T1-LABEL: strb_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, #255
; CHECK-T1-NEXT:    strb r1, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: strb_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    strb r1, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %conv = trunc i32 %x to i8
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  store i8 %conv, i8* %add.ptr, align 1
  ret void
}

define void @strh_ri_negative255(i8* %p, i32 %x) {
; CHECK-T1-LABEL: strh_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, #255
; CHECK-T1-NEXT:    strh r1, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: strh_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    strh r1, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %conv = trunc i32 %x to i16
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  %0 = bitcast i8* %add.ptr to i16*
  store i16 %conv, i16* %0, align 2
  ret void
}

define void @str_ri_negative255(i8* %p, i32 %x) {
; CHECK-T1-LABEL: str_ri_negative255:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    subs r0, #255
; CHECK-T1-NEXT:    str r1, [r0]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: str_ri_negative255:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    str r1, [r0, #-255]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -255
  %0 = bitcast i8* %add.ptr to i32*
  store i32 %x, i32* %0, align 4
  ret void
}


; Negative 256 offset

define i32 @ldrsb_ri_negative256(i8* %p) {
; CHECK-T1-LABEL: ldrsb_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #255
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrsb r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrsb_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r1, #255
; CHECK-T2-NEXT:    ldrsb r0, [r0, r1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  %0 = load i8, i8* %add.ptr, align 1
  %conv = sext i8 %0 to i32
  ret i32 %conv
}

define i32 @ldrsh_ri_negative256(i8* %p) {
; CHECK-T1-LABEL: ldrsh_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #255
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrsh r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrsh_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r1, #255
; CHECK-T2-NEXT:    ldrsh r0, [r0, r1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = sext i16 %1 to i32
  ret i32 %conv
}

define i32 @ldrb_ri_negative256(i8* %p) {
; CHECK-T1-LABEL: ldrb_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #255
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrb r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrb_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r1, #255
; CHECK-T2-NEXT:    ldrb r0, [r0, r1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  %0 = load i8, i8* %add.ptr, align 1
  %conv = zext i8 %0 to i32
  ret i32 %conv
}

define i32 @ldrh_ri_negative256(i8* %p) {
; CHECK-T1-LABEL: ldrh_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #255
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldrh r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldrh_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r1, #255
; CHECK-T2-NEXT:    ldrh r0, [r0, r1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  %0 = bitcast i8* %add.ptr to i16*
  %1 = load i16, i16* %0, align 2
  %conv = zext i16 %1 to i32
  ret i32 %conv
}

define i32 @ldr_ri_negative256(i8* %p) {
; CHECK-T1-LABEL: ldr_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r1, #255
; CHECK-T1-NEXT:    mvns r1, r1
; CHECK-T1-NEXT:    ldr r0, [r0, r1]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: ldr_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r1, #255
; CHECK-T2-NEXT:    ldr r0, [r0, r1]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  %0 = bitcast i8* %add.ptr to i32*
  %1 = load i32, i32* %0, align 4
  ret i32 %1
}

define void @strb_ri_negative256(i8* %p, i32 %x) {
; CHECK-T1-LABEL: strb_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r2, #255
; CHECK-T1-NEXT:    mvns r2, r2
; CHECK-T1-NEXT:    strb r1, [r0, r2]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: strb_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r2, #255
; CHECK-T2-NEXT:    strb r1, [r0, r2]
; CHECK-T2-NEXT:    bx lr
entry:
  %conv = trunc i32 %x to i8
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  store i8 %conv, i8* %add.ptr, align 1
  ret void
}

define void @strh_ri_negative256(i8* %p, i32 %x) {
; CHECK-T1-LABEL: strh_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r2, #255
; CHECK-T1-NEXT:    mvns r2, r2
; CHECK-T1-NEXT:    strh r1, [r0, r2]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: strh_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r2, #255
; CHECK-T2-NEXT:    strh r1, [r0, r2]
; CHECK-T2-NEXT:    bx lr
entry:
  %conv = trunc i32 %x to i16
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  %0 = bitcast i8* %add.ptr to i16*
  store i16 %conv, i16* %0, align 2
  ret void
}

define void @str_ri_negative256(i8* %p, i32 %x) {
; CHECK-T1-LABEL: str_ri_negative256:
; CHECK-T1:       @ %bb.0: @ %entry
; CHECK-T1-NEXT:    movs r2, #255
; CHECK-T1-NEXT:    mvns r2, r2
; CHECK-T1-NEXT:    str r1, [r0, r2]
; CHECK-T1-NEXT:    bx lr
;
; CHECK-T2-LABEL: str_ri_negative256:
; CHECK-T2:       @ %bb.0: @ %entry
; CHECK-T2-NEXT:    mvn r2, #255
; CHECK-T2-NEXT:    str r1, [r0, r2]
; CHECK-T2-NEXT:    bx lr
entry:
  %add.ptr = getelementptr inbounds i8, i8* %p, i32 -256
  %0 = bitcast i8* %add.ptr to i32*
  store i32 %x, i32* %0, align 4
  ret void
}
