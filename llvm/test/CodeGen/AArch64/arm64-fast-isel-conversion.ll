; RUN: llc -O0 -fast-isel -fast-isel-abort=1 -verify-machineinstrs -mtriple=arm64-apple-darwin -mcpu=cyclone < %s | FileCheck %s

;; Test various conversions.
define zeroext i32 @trunc_(i8 zeroext %a, i16 zeroext %b, i32 %c, i64 %d) nounwind ssp {
entry:
; CHECK-LABEL: trunc_
; CHECK: sub sp, sp, #16
; CHECK: strb w0, [sp, #15]
; CHECK: strh w1, [sp, #12]
; CHECK: str w2, [sp, #8]
; CHECK: str x3, [sp]
; CHECK: ldr x3, [sp]
; CHECK: mov x0, x3
; CHECK: str w0, [sp, #8]
; CHECK: ldr w0, [sp, #8]
; CHECK: strh w0, [sp, #12]
; CHECK: ldrh w0, [sp, #12]
; CHECK: strb w0, [sp, #15]
; CHECK: ldrb w0, [sp, #15]
; CHECK: add sp, sp, #16
; CHECK: ret
  %a.addr = alloca i8, align 1
  %b.addr = alloca i16, align 2
  %c.addr = alloca i32, align 4
  %d.addr = alloca i64, align 8
  store i8 %a, i8* %a.addr, align 1
  store i16 %b, i16* %b.addr, align 2
  store i32 %c, i32* %c.addr, align 4
  store i64 %d, i64* %d.addr, align 8
  %tmp = load i64, i64* %d.addr, align 8
  %conv = trunc i64 %tmp to i32
  store i32 %conv, i32* %c.addr, align 4
  %tmp1 = load i32, i32* %c.addr, align 4
  %conv2 = trunc i32 %tmp1 to i16
  store i16 %conv2, i16* %b.addr, align 2
  %tmp3 = load i16, i16* %b.addr, align 2
  %conv4 = trunc i16 %tmp3 to i8
  store i8 %conv4, i8* %a.addr, align 1
  %tmp5 = load i8, i8* %a.addr, align 1
  %conv6 = zext i8 %tmp5 to i32
  ret i32 %conv6
}

define i64 @zext_(i8 zeroext %a, i16 zeroext %b, i32 %c, i64 %d) nounwind ssp {
entry:
; CHECK-LABEL: zext_
; CHECK: sub sp, sp, #16
; CHECK: strb w0, [sp, #15]
; CHECK: strh w1, [sp, #12]
; CHECK: str w2, [sp, #8]
; CHECK: str x3, [sp]
; CHECK: ldrb w0, [sp, #15]
; CHECK: strh w0, [sp, #12]
; CHECK: ldrh w0, [sp, #12]
; CHECK: str w0, [sp, #8]
; CHECK: ldr w0, [sp, #8]
; CHECK: mov x3, x0
; CHECK: str x3, [sp]
; CHECK: ldr x0, [sp]
; CHECK: ret
  %a.addr = alloca i8, align 1
  %b.addr = alloca i16, align 2
  %c.addr = alloca i32, align 4
  %d.addr = alloca i64, align 8
  store i8 %a, i8* %a.addr, align 1
  store i16 %b, i16* %b.addr, align 2
  store i32 %c, i32* %c.addr, align 4
  store i64 %d, i64* %d.addr, align 8
  %tmp = load i8, i8* %a.addr, align 1
  %conv = zext i8 %tmp to i16
  store i16 %conv, i16* %b.addr, align 2
  %tmp1 = load i16, i16* %b.addr, align 2
  %conv2 = zext i16 %tmp1 to i32
  store i32 %conv2, i32* %c.addr, align 4
  %tmp3 = load i32, i32* %c.addr, align 4
  %conv4 = zext i32 %tmp3 to i64
  store i64 %conv4, i64* %d.addr, align 8
  %tmp5 = load i64, i64* %d.addr, align 8
  ret i64 %tmp5
}

define i32 @zext_i1_i32(i1 zeroext %a) nounwind ssp {
entry:
; CHECK-LABEL: zext_i1_i32
; CHECK-NOT:   and w0, w0, #0x1
; CHECK:       ret
  %conv = zext i1 %a to i32
  ret i32 %conv;
}

define i64 @zext_i1_i64(i1 zeroext %a) nounwind ssp {
entry:
; CHECK-LABEL: zext_i1_i64
; CHECK-NOT:   and w0, w0, #0x1
; CHECK:       ret
  %conv = zext i1 %a to i64
  ret i64 %conv;
}

define i64 @sext_(i8 signext %a, i16 signext %b, i32 %c, i64 %d) nounwind ssp {
entry:
; CHECK-LABEL: sext_
; CHECK: sub sp, sp, #16
; CHECK: strb w0, [sp, #15]
; CHECK: strh w1, [sp, #12]
; CHECK: str w2, [sp, #8]
; CHECK: str x3, [sp]
; CHECK: ldrsb w0, [sp, #15]
; CHECK: strh w0, [sp, #12]
; CHECK: ldrsh w0, [sp, #12]
; CHECK: str w0, [sp, #8]
; CHECK: ldrsw x3, [sp, #8]
; CHECK: str x3, [sp]
; CHECK: ldr x0, [sp]
; CHECK: ret
  %a.addr = alloca i8, align 1
  %b.addr = alloca i16, align 2
  %c.addr = alloca i32, align 4
  %d.addr = alloca i64, align 8
  store i8 %a, i8* %a.addr, align 1
  store i16 %b, i16* %b.addr, align 2
  store i32 %c, i32* %c.addr, align 4
  store i64 %d, i64* %d.addr, align 8
  %tmp = load i8, i8* %a.addr, align 1
  %conv = sext i8 %tmp to i16
  store i16 %conv, i16* %b.addr, align 2
  %tmp1 = load i16, i16* %b.addr, align 2
  %conv2 = sext i16 %tmp1 to i32
  store i32 %conv2, i32* %c.addr, align 4
  %tmp3 = load i32, i32* %c.addr, align 4
  %conv4 = sext i32 %tmp3 to i64
  store i64 %conv4, i64* %d.addr, align 8
  %tmp5 = load i64, i64* %d.addr, align 8
  ret i64 %tmp5
}

; Test sext i8 to i64

define zeroext i64 @sext_i8_i64(i8 zeroext %in) {
; CHECK-LABEL: sext_i8_i64:
; CHECK: mov x[[TMP:[0-9]+]], x0
; CHECK: sxtb x0, w[[TMP]]
  %big = sext i8 %in to i64
  ret i64 %big
}

define zeroext i64 @sext_i16_i64(i16 zeroext %in) {
; CHECK-LABEL: sext_i16_i64:
; CHECK: mov x[[TMP:[0-9]+]], x0
; CHECK: sxth x0, w[[TMP]]
  %big = sext i16 %in to i64
  ret i64 %big
}

; Test sext i1 to i32
define i32 @sext_i1_i32(i1 signext %a) nounwind ssp {
entry:
; CHECK-LABEL: sext_i1_i32
; CHECK-NOT:   sbfx w0, w0, #0, #1
; CHECK:       ret
  %conv = sext i1 %a to i32
  ret i32 %conv
}

; Test sext i1 to i16
define signext i16 @sext_i1_i16(i1 %a) nounwind ssp {
entry:
; CHECK-LABEL: sext_i1_i16
; CHECK: sbfx w0, w0, #0, #1
  %conv = sext i1 %a to i16
  ret i16 %conv
}

; Test sext i1 to i8
define signext i8 @sext_i1_i8(i1 %a) nounwind ssp {
entry:
; CHECK-LABEL: sext_i1_i8
; CHECK: sbfx w0, w0, #0, #1
  %conv = sext i1 %a to i8
  ret i8 %conv
}

; Test fpext
define double @fpext_(float %a) nounwind ssp {
entry:
; CHECK-LABEL: fpext_
; CHECK: fcvt d0, s0
  %conv = fpext float %a to double
  ret double %conv
}

; Test fptrunc
define float @fptrunc_(double %a) nounwind ssp {
entry:
; CHECK-LABEL: fptrunc_
; CHECK: fcvt s0, d0
  %conv = fptrunc double %a to float
  ret float %conv
}

; Test fptosi
define i32 @fptosi_ws(float %a) nounwind ssp {
entry:
; CHECK-LABEL: fptosi_ws
; CHECK: fcvtzs w0, s0
  %conv = fptosi float %a to i32
  ret i32 %conv
}

; Test fptosi
define i32 @fptosi_wd(double %a) nounwind ssp {
entry:
; CHECK-LABEL: fptosi_wd
; CHECK: fcvtzs w0, d0
  %conv = fptosi double %a to i32
  ret i32 %conv
}

; Test fptoui
define i32 @fptoui_ws(float %a) nounwind ssp {
entry:
; CHECK-LABEL: fptoui_ws
; CHECK: fcvtzu w0, s0
  %conv = fptoui float %a to i32
  ret i32 %conv
}

; Test fptoui
define i32 @fptoui_wd(double %a) nounwind ssp {
entry:
; CHECK-LABEL: fptoui_wd
; CHECK: fcvtzu w0, d0
  %conv = fptoui double %a to i32
  ret i32 %conv
}

; Test sitofp
define float @sitofp_sw_i1(i1 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_sw_i1
; CHECK: sbfx w0, w0, #0, #1
; CHECK: scvtf s0, w0
  %conv = sitofp i1 %a to float
  ret float %conv
}

; Test sitofp
define float @sitofp_sw_i8(i8 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_sw_i8
; CHECK: sxtb w0, w0
; CHECK: scvtf s0, w0
  %conv = sitofp i8 %a to float
  ret float %conv
}

; Test sitofp
define float @sitofp_sw_i16(i16 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_sw_i16
  %conv = sitofp i16 %a to float
  ret float %conv
}

; Test sitofp
define float @sitofp_sw(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_sw
; CHECK: scvtf s0, w0
  %conv = sitofp i32 %a to float
  ret float %conv
}

; Test sitofp
define float @sitofp_sx(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_sx
; CHECK: scvtf s0, x0
  %conv = sitofp i64 %a to float
  ret float %conv
}

; Test sitofp
define double @sitofp_dw(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_dw
; CHECK: scvtf d0, w0
  %conv = sitofp i32 %a to double
  ret double %conv
}

; Test sitofp
define double @sitofp_dx(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: sitofp_dx
; CHECK: scvtf d0, x0
  %conv = sitofp i64 %a to double
  ret double %conv
}

; Test uitofp
define float @uitofp_sw_i1(i1 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_sw_i1
; CHECK: and w0, w0, #0x1
; CHECK: ucvtf s0, w0
  %conv = uitofp i1 %a to float
  ret float %conv
}

; Test uitofp
define float @uitofp_sw_i8(i8 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_sw_i8
  %conv = uitofp i8 %a to float
  ret float %conv
}

; Test uitofp
define float @uitofp_sw_i16(i16 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_sw_i16
  %conv = uitofp i16 %a to float
  ret float %conv
}

; Test uitofp
define float @uitofp_sw(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_sw
; CHECK: ucvtf s0, w0
  %conv = uitofp i32 %a to float
  ret float %conv
}

; Test uitofp
define float @uitofp_sx(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_sx
; CHECK: ucvtf s0, x0
  %conv = uitofp i64 %a to float
  ret float %conv
}

; Test uitofp
define double @uitofp_dw(i32 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_dw
; CHECK: ucvtf d0, w0
  %conv = uitofp i32 %a to double
  ret double %conv
}

; Test uitofp
define double @uitofp_dx(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: uitofp_dx
; CHECK: ucvtf d0, x0
  %conv = uitofp i64 %a to double
  ret double %conv
}

define i32 @i64_trunc_i32(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: i64_trunc_i32
; CHECK: mov x1, x0
  %conv = trunc i64 %a to i32
  ret i32 %conv
}

define zeroext i16 @i64_trunc_i16(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: i64_trunc_i16
; CHECK: mov x[[REG:[0-9]+]], x0
; CHECK: and [[REG2:w[0-9]+]], w[[REG]], #0xffff
; CHECK: uxth w0, [[REG2]]
  %conv = trunc i64 %a to i16
  ret i16 %conv
}

define zeroext i8 @i64_trunc_i8(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: i64_trunc_i8
; CHECK: mov x[[REG:[0-9]+]], x0
; CHECK: and [[REG2:w[0-9]+]], w[[REG]], #0xff
; CHECK: uxtb w0, [[REG2]]
  %conv = trunc i64 %a to i8
  ret i8 %conv
}

define zeroext i1 @i64_trunc_i1(i64 %a) nounwind ssp {
entry:
; CHECK-LABEL: i64_trunc_i1
; CHECK: mov x[[REG:[0-9]+]], x0
; CHECK: and [[REG2:w[0-9]+]], w[[REG]], #0x1
; CHECK: and w0, [[REG2]], #0x1
  %conv = trunc i64 %a to i1
  ret i1 %conv
}

; rdar://15101939
define void @stack_trunc() nounwind {
; CHECK-LABEL: stack_trunc
; CHECK: sub  sp, sp, #16
; CHECK: ldr  [[REG:x[0-9]+]], [sp]
; CHECK: mov  x[[REG2:[0-9]+]], [[REG]]
; CHECK: and  [[REG3:w[0-9]+]], w[[REG2]], #0xff
; CHECK: strb [[REG3]], [sp, #15]
; CHECK: add  sp, sp, #16
  %a = alloca i8, align 1
  %b = alloca i64, align 8
  %c = load i64, i64* %b, align 8
  %d = trunc i64 %c to i8
  store i8 %d, i8* %a, align 1
  ret void
}

define zeroext i64 @zext_i8_i64(i8 zeroext %in) {
; CHECK-LABEL: zext_i8_i64:
; CHECK-NOT:   ubfx x0, {{x[0-9]+}}, #0, #8
; CHECK:       ret
  %big = zext i8 %in to i64
  ret i64 %big
}
define zeroext i64 @zext_i16_i64(i16 zeroext %in) {
; CHECK-LABEL: zext_i16_i64:
; CHECK-NOT:   ubfx x0, {{x[0-9]+}}, #0, #16
; CHECK:       ret
  %big = zext i16 %in to i64
  ret i64 %big
}

define float @bitcast_i32_to_float(i32 %a) {
  %1 = bitcast i32 %a to float
  ret float %1
}

define double @bitcast_i64_to_double(i64 %a) {
  %1 = bitcast i64 %a to double
  ret double %1
}

define i32 @bitcast_float_to_i32(float %a) {
  %1 = bitcast float %a to i32
  ret i32 %1
}

define i64 @bitcast_double_to_i64(double %a) {
  %1 = bitcast double %a to i64
  ret i64 %1
}

