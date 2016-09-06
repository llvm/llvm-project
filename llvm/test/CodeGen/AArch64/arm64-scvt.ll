; RUN: llc < %s -mtriple=arm64-eabi -mcpu=cyclone -aarch64-neon-syntax=apple | FileCheck %s
; RUN: llc < %s -mtriple=arm64-eabi -mcpu=cortex-a57 | FileCheck --check-prefix=CHECK-A57 %s
; rdar://13082402

define float @t1(i32* nocapture %src) nounwind ssp {
entry:
; CHECK-LABEL: t1:
; CHECK: ldr s0, [x0]
; CHECK: scvtf s0, s0
  %tmp1 = load i32, i32* %src, align 4
  %tmp2 = sitofp i32 %tmp1 to float
  ret float %tmp2
}

define float @t2(i32* nocapture %src) nounwind ssp {
entry:
; CHECK-LABEL: t2:
; CHECK: ldr s0, [x0]
; CHECK: ucvtf s0, s0
  %tmp1 = load i32, i32* %src, align 4
  %tmp2 = uitofp i32 %tmp1 to float
  ret float %tmp2
}

define double @t3(i64* nocapture %src) nounwind ssp {
entry:
; CHECK-LABEL: t3:
; CHECK: ldr d0, [x0]
; CHECK: scvtf d0, d0
  %tmp1 = load i64, i64* %src, align 4
  %tmp2 = sitofp i64 %tmp1 to double
  ret double %tmp2
}

define double @t4(i64* nocapture %src) nounwind ssp {
entry:
; CHECK-LABEL: t4:
; CHECK: ldr d0, [x0]
; CHECK: ucvtf d0, d0
  %tmp1 = load i64, i64* %src, align 4
  %tmp2 = uitofp i64 %tmp1 to double
  ret double %tmp2
}

; rdar://13136456
define double @t5(i32* nocapture %src) nounwind ssp optsize {
entry:
; CHECK-LABEL: t5:
; CHECK: ldr [[REG:w[0-9]+]], [x0]
; CHECK: scvtf d0, [[REG]]
  %tmp1 = load i32, i32* %src, align 4
  %tmp2 = sitofp i32 %tmp1 to double
  ret double %tmp2
}

; Check that we load in FP register when we want to convert into
; floating point value.
; This is much faster than loading on GPR and making the conversion
; GPR -> FPR.
; <rdar://problem/14599607>
;
; Check the flollowing patterns for signed/unsigned:
; 1. load with scaled imm to float.
; 2. load with scaled register to float.
; 3. load with scaled imm to double.
; 4. load with scaled register to double.
; 5. load with unscaled imm to float.
; 6. load with unscaled imm to double.
; With loading size: 8, 16, 32, and 64-bits.

; ********* 1. load with scaled imm to float. *********
define float @fct1(i8* nocapture %sp0) {
; CHECK-LABEL: fct1:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 1
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = uitofp i8 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @fct2(i16* nocapture %sp0) {
; CHECK-LABEL: fct2:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, #2]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 1
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = uitofp i16 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @fct3(i32* nocapture %sp0) {
; CHECK-LABEL: fct3:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, #4]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 1
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = uitofp i32 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; i64 -> f32 is not supported on floating point unit.
define float @fct4(i64* nocapture %sp0) {
; CHECK-LABEL: fct4:
; CHECK: ldr x[[REGNUM:[0-9]+]], [x0, #8]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], x[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 1
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = uitofp i64 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; ********* 2. load with scaled register to float. *********
define float @fct5(i8* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct5:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, x1]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = uitofp i8 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @fct6(i16* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct6:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = uitofp i16 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @fct7(i32* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct7:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, x1, lsl #2]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = uitofp i32 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; i64 -> f32 is not supported on floating point unit.
define float @fct8(i64* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct8:
; CHECK: ldr x[[REGNUM:[0-9]+]], [x0, x1, lsl #3]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], x[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = uitofp i64 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}


; ********* 3. load with scaled imm to double. *********
define double @fct9(i8* nocapture %sp0) {
; CHECK-LABEL: fct9:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 1
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = uitofp i8 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct10(i16* nocapture %sp0) {
; CHECK-LABEL: fct10:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, #2]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 1
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = uitofp i16 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct11(i32* nocapture %sp0) {
; CHECK-LABEL: fct11:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, #4]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 1
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = uitofp i32 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct12(i64* nocapture %sp0) {
; CHECK-LABEL: fct12:
; CHECK: ldr d[[REGNUM:[0-9]+]], [x0, #8]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 1
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = uitofp i64 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

; ********* 4. load with scaled register to double. *********
define double @fct13(i8* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct13:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, x1]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = uitofp i8 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct14(i16* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct14:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = uitofp i16 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct15(i32* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct15:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, x1, lsl #2]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = uitofp i32 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct16(i64* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: fct16:
; CHECK: ldr d[[REGNUM:[0-9]+]], [x0, x1, lsl #3]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = uitofp i64 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

; ********* 5. load with unscaled imm to float. *********
define float @fct17(i8* nocapture %sp0) {
entry:
; CHECK-LABEL: fct17:
; CHECK: ldur b[[REGNUM:[0-9]+]], [x0, #-1]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i8* %sp0 to i64
  %add = add i64 %bitcast, -1
  %addr = inttoptr i64 %add to i8*
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = uitofp i8 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @fct18(i16* nocapture %sp0) {
; CHECK-LABEL: fct18:
; CHECK: ldur h[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i16* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i16*
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = uitofp i16 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @fct19(i32* nocapture %sp0) {
; CHECK-LABEL: fct19:
; CHECK: ldur s[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], s[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i32* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i32*
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = uitofp i32 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; i64 -> f32 is not supported on floating point unit.
define float @fct20(i64* nocapture %sp0) {
; CHECK-LABEL: fct20:
; CHECK: ldur x[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:s[0-9]+]], x[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i64* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i64*
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = uitofp i64 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i

}

; ********* 6. load with unscaled imm to double. *********
define double @fct21(i8* nocapture %sp0) {
entry:
; CHECK-LABEL: fct21:
; CHECK: ldur b[[REGNUM:[0-9]+]], [x0, #-1]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i8* %sp0 to i64
  %add = add i64 %bitcast, -1
  %addr = inttoptr i64 %add to i8*
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = uitofp i8 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct22(i16* nocapture %sp0) {
; CHECK-LABEL: fct22:
; CHECK: ldur h[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i16* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i16*
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = uitofp i16 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct23(i32* nocapture %sp0) {
; CHECK-LABEL: fct23:
; CHECK: ldur s[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i32* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i32*
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = uitofp i32 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @fct24(i64* nocapture %sp0) {
; CHECK-LABEL: fct24:
; CHECK: ldur d[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: ucvtf [[REG:d[0-9]+]], d[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i64* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i64*
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = uitofp i64 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i

}

; ********* 1s. load with scaled imm to float. *********
define float @sfct1(i8* nocapture %sp0) {
; CHECK-LABEL: sfct1:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: sshll.8h [[SEXTREG1:v[0-9]+]], v[[REGNUM]], #0
; CHECK-NEXT: sshll.4s v[[SEXTREG:[0-9]+]], [[SEXTREG1]], #0
; CHECK: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
; CHECK-A57-LABEL: sfct1:
; CHECK-A57: ldrsb w[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-A57-NEXT: scvtf [[REG:s[0-9]+]], w[[REGNUM]]
; CHECK-A57-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 1
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = sitofp i8 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @sfct2(i16* nocapture %sp0) {
; CHECK-LABEL: sfct2:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, #2]
; CHECK-NEXT: sshll.4s v[[SEXTREG:[0-9]+]], v[[REGNUM]], #0
; CHECK: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 1
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = sitofp i16 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @sfct3(i32* nocapture %sp0) {
; CHECK-LABEL: sfct3:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, #4]
; CHECK-NEXT: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 1
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = sitofp i32 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; i64 -> f32 is not supported on floating point unit.
define float @sfct4(i64* nocapture %sp0) {
; CHECK-LABEL: sfct4:
; CHECK: ldr x[[REGNUM:[0-9]+]], [x0, #8]
; CHECK-NEXT: scvtf [[REG:s[0-9]+]], x[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 1
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = sitofp i64 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; ********* 2s. load with scaled register to float. *********
define float @sfct5(i8* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct5:
; CHECK: ldr b[[REGNUM:[0-9]+]], [x0, x1]
; CHECK-NEXT: sshll.8h [[SEXTREG1:v[0-9]+]], v[[REGNUM]], #0
; CHECK-NEXT: sshll.4s v[[SEXTREG:[0-9]+]], [[SEXTREG1]], #0
; CHECK: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
; CHECK-A57-LABEL: sfct5:
; CHECK-A57: ldrsb w[[REGNUM:[0-9]+]], [x0, x1]
; CHECK-A57-NEXT: scvtf [[REG:s[0-9]+]], w[[REGNUM]]
; CHECK-A57-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = sitofp i8 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @sfct6(i16* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct6:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: sshll.4s v[[SEXTREG:[0-9]+]], v[[REGNUM]], #0
; CHECK: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = sitofp i16 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @sfct7(i32* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct7:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, x1, lsl #2]
; CHECK-NEXT: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = sitofp i32 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; i64 -> f32 is not supported on floating point unit.
define float @sfct8(i64* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct8:
; CHECK: ldr x[[REGNUM:[0-9]+]], [x0, x1, lsl #3]
; CHECK-NEXT: scvtf [[REG:s[0-9]+]], x[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = sitofp i64 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; ********* 3s. load with scaled imm to double. *********
define double @sfct9(i8* nocapture %sp0) {
; CHECK-LABEL: sfct9:
; CHECK: ldrsb w[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: scvtf [[REG:d[0-9]+]], w[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 1
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = sitofp i8 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct10(i16* nocapture %sp0) {
; CHECK-LABEL: sfct10:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, #2]
; CHECK-NEXT: sshll.4s [[SEXTREG1:v[0-9]+]], v[[REGNUM]], #0
; CHECK-NEXT: sshll.2d v[[SEXTREG:[0-9]+]], [[SEXTREG1]], #0
; CHECK: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
; CHECK-A57-LABEL: sfct10:
; CHECK-A57: ldrsh w[[REGNUM:[0-9]+]], [x0, #2]
; CHECK-A57-NEXT: scvtf [[REG:d[0-9]+]], w[[REGNUM]]
; CHECK-A57-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 1
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = sitofp i16 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct11(i32* nocapture %sp0) {
; CHECK-LABEL: sfct11:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, #4]
; CHECK-NEXT: sshll.2d v[[SEXTREG:[0-9]+]], v[[REGNUM]], #0
; CHECK: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 1
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = sitofp i32 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct12(i64* nocapture %sp0) {
; CHECK-LABEL: sfct12:
; CHECK: ldr d[[REGNUM:[0-9]+]], [x0, #8]
; CHECK-NEXT: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 1
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = sitofp i64 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

; ********* 4s. load with scaled register to double. *********
define double @sfct13(i8* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct13:
; CHECK: ldrsb w[[REGNUM:[0-9]+]], [x0, x1]
; CHECK-NEXT: scvtf [[REG:d[0-9]+]], w[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i8, i8* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = sitofp i8 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct14(i16* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct14:
; CHECK: ldr h[[REGNUM:[0-9]+]], [x0, x1, lsl #1]
; CHECK-NEXT: sshll.4s [[SEXTREG1:v[0-9]+]], v[[REGNUM]], #0
; CHECK-NEXT: sshll.2d v[[SEXTREG:[0-9]+]], [[SEXTREG1]], #0
; CHECK: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
; CHECK-A57-LABEL: sfct14:
; CHECK-A57: ldrsh w[[REGNUM:[0-9]+]], [x0, x1, lsl #1]
; CHECK-A57-NEXT: scvtf [[REG:d[0-9]+]], w[[REGNUM]]
; CHECK-A57-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i16, i16* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = sitofp i16 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct15(i32* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct15:
; CHECK: ldr s[[REGNUM:[0-9]+]], [x0, x1, lsl #2]
; CHECK-NEXT: sshll.2d v[[SEXTREG:[0-9]+]], v[[REGNUM]], #0
; CHECK: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = sitofp i32 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct16(i64* nocapture %sp0, i64 %offset) {
; CHECK-LABEL: sfct16:
; CHECK: ldr d[[REGNUM:[0-9]+]], [x0, x1, lsl #3]
; CHECK-NEXT: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i64, i64* %sp0, i64 %offset
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = sitofp i64 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

; ********* 5s. load with unscaled imm to float. *********
define float @sfct17(i8* nocapture %sp0) {
entry:
; CHECK-LABEL: sfct17:
; CHECK: ldur b[[REGNUM:[0-9]+]], [x0, #-1]
; CHECK-NEXT: sshll.8h [[SEXTREG1:v[0-9]+]], v[[REGNUM]], #0
; CHECK-NEXT: sshll.4s v[[SEXTREG:[0-9]+]], [[SEXTREG1]], #0
; CHECK: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
; CHECK-A57-LABEL: sfct17:
; CHECK-A57: ldursb w[[REGNUM:[0-9]+]], [x0, #-1]
; CHECK-A57-NEXT: scvtf [[REG:s[0-9]+]], w[[REGNUM]]
; CHECK-A57-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i8* %sp0 to i64
  %add = add i64 %bitcast, -1
  %addr = inttoptr i64 %add to i8*
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = sitofp i8 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @sfct18(i16* nocapture %sp0) {
; CHECK-LABEL: sfct18:
; CHECK: ldur h[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: sshll.4s v[[SEXTREG:[0-9]+]], v[[REGNUM]], #0
; CHECK: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i16* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i16*
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = sitofp i16 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define float @sfct19(i32* nocapture %sp0) {
; CHECK-LABEL: sfct19:
; CHECK: ldur s[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: scvtf [[REG:s[0-9]+]], s[[SEXTREG]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i32* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i32*
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = sitofp i32 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

; i64 -> f32 is not supported on floating point unit.
define float @sfct20(i64* nocapture %sp0) {
; CHECK-LABEL: sfct20:
; CHECK: ldur x[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: scvtf [[REG:s[0-9]+]], x[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i64* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i64*
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = sitofp i64 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i

}

; ********* 6s. load with unscaled imm to double. *********
define double @sfct21(i8* nocapture %sp0) {
entry:
; CHECK-LABEL: sfct21:
; CHECK: ldursb w[[REGNUM:[0-9]+]], [x0, #-1]
; CHECK-NEXT: scvtf [[REG:d[0-9]+]], w[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i8* %sp0 to i64
  %add = add i64 %bitcast, -1
  %addr = inttoptr i64 %add to i8*
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = sitofp i8 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct22(i16* nocapture %sp0) {
; CHECK-LABEL: sfct22:
; CHECK: ldur h[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: sshll.4s [[SEXTREG1:v[0-9]+]], v[[REGNUM]], #0
; CHECK-NEXT: sshll.2d v[[SEXTREG:[0-9]+]], [[SEXTREG1]], #0
; CHECK: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
; CHECK-A57-LABEL: sfct22:
; CHECK-A57: ldursh w[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-A57-NEXT: scvtf [[REG:d[0-9]+]], w[[REGNUM]]
; CHECK-A57-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i16* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i16*
  %pix_sp0.0.copyload = load i16, i16* %addr, align 1
  %val = sitofp i16 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct23(i32* nocapture %sp0) {
; CHECK-LABEL: sfct23:
; CHECK: ldur s[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: sshll.2d v[[SEXTREG:[0-9]+]], v[[REGNUM]], #0
; CHECK: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i32* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i32*
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = sitofp i32 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

define double @sfct24(i64* nocapture %sp0) {
; CHECK-LABEL: sfct24:
; CHECK: ldur d[[REGNUM:[0-9]+]], [x0, #1]
; CHECK-NEXT: scvtf [[REG:d[0-9]+]], d[[SEXTREG]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
  %bitcast = ptrtoint i64* %sp0 to i64
  %add = add i64 %bitcast, 1
  %addr = inttoptr i64 %add to i64*
  %pix_sp0.0.copyload = load i64, i64* %addr, align 1
  %val = sitofp i64 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i

}

; Check that we do not use SSHLL code sequence when code size is a concern.
define float @codesize_sfct17(i8* nocapture %sp0) optsize {
entry:
; CHECK-LABEL: codesize_sfct17:
; CHECK: ldursb w[[REGNUM:[0-9]+]], [x0, #-1]
; CHECK-NEXT: scvtf [[REG:s[0-9]+]], w[[REGNUM]]
; CHECK-NEXT: fmul s0, [[REG]], [[REG]]
  %bitcast = ptrtoint i8* %sp0 to i64
  %add = add i64 %bitcast, -1
  %addr = inttoptr i64 %add to i8*
  %pix_sp0.0.copyload = load i8, i8* %addr, align 1
  %val = sitofp i8 %pix_sp0.0.copyload to float
  %vmull.i = fmul float %val, %val
  ret float %vmull.i
}

define double @codesize_sfct11(i32* nocapture %sp0) minsize {
; CHECK-LABEL: sfct11:
; CHECK: ldr w[[REGNUM:[0-9]+]], [x0, #4]
; CHECK-NEXT: scvtf [[REG:d[0-9]+]], w[[REGNUM]]
; CHECK-NEXT: fmul d0, [[REG]], [[REG]]
entry:
  %addr = getelementptr i32, i32* %sp0, i64 1
  %pix_sp0.0.copyload = load i32, i32* %addr, align 1
  %val = sitofp i32 %pix_sp0.0.copyload to double
  %vmull.i = fmul double %val, %val
  ret double %vmull.i
}

; Adding fp128 custom lowering makes these a little fragile since we have to
; return the correct mix of Legal/Expand from the custom method.
;
; rdar://problem/14991489

define float @float_from_i128(i128 %in) {
; CHECK-LABEL: float_from_i128:
; CHECK: bl {{_?__floatuntisf}}
  %conv = uitofp i128 %in to float
  ret float %conv
}

define double @double_from_i128(i128 %in) {
; CHECK-LABEL: double_from_i128:
; CHECK: bl {{_?__floattidf}}
  %conv = sitofp i128 %in to double
  ret double %conv
}

define fp128 @fp128_from_i128(i128 %in) {
; CHECK-LABEL: fp128_from_i128:
; CHECK: bl {{_?__floatuntitf}}
  %conv = uitofp i128 %in to fp128
  ret fp128 %conv
}

define i128 @i128_from_float(float %in) {
; CHECK-LABEL: i128_from_float
; CHECK: bl {{_?__fixsfti}}
  %conv = fptosi float %in to i128
  ret i128 %conv
}

define i128 @i128_from_double(double %in) {
; CHECK-LABEL: i128_from_double
; CHECK: bl {{_?__fixunsdfti}}
  %conv = fptoui double %in to i128
  ret i128 %conv
}

define i128 @i128_from_fp128(fp128 %in) {
; CHECK-LABEL: i128_from_fp128
; CHECK: bl {{_?__fixtfti}}
  %conv = fptosi fp128 %in to i128
  ret i128 %conv
}

