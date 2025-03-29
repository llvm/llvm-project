; RUN: opt < %s -mtriple=aarch64--linux-gnu -passes="print<cost-model>" 2>&1 -disable-output | FileCheck %s --check-prefixes=COST,COST-NOFP16
; RUN: opt < %s -mtriple=aarch64--linux-gnu -passes="print<cost-model>" 2>&1 -disable-output -mattr=+fullfp16 | FileCheck %s --check-prefixes=COST,COST-FULLFP16
; RUN: llc < %s -mtriple=aarch64--linux-gnu -mattr=+fullfp16 | FileCheck %s --check-prefix=CODE

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

; COST-LABEL: v8i8_select_eq
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = icmp eq <8 x i8> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x i8> %a, <8 x i8> %c

; CODE-LABEL: v8i8_select_eq
; CODE:       bb.0
; CODE-NEXT:    cmeq  v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret

define <8 x i8> @v8i8_select_eq(<8 x i8> %a, <8 x i8> %b, <8 x i8> %c) {
  %cmp.1 = icmp eq <8 x i8> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x i8> %a, <8 x i8> %c
  ret <8 x i8> %s.1
}

; COST-LABEL: v16i8_select_sgt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = icmp sgt <16 x i8> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <16 x i1> %cmp.1, <16 x i8> %a, <16 x i8> %c

; CODE-LABEL: v16i8_select_sgt
; CODE:       bb.0
; CODE-NEXT:    cmgt  v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

define <16 x i8> @v16i8_select_sgt(<16 x i8> %a, <16 x i8> %b, <16 x i8> %c) {
  %cmp.1 = icmp sgt <16 x i8> %a, %b
  %s.1 = select <16 x i1> %cmp.1, <16 x i8> %a, <16 x i8> %c
  ret <16 x i8> %s.1
}

; COST-LABEL: v4i16_select_ne
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = icmp ne <4 x i16> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x i16> %a, <4 x i16> %c

; CODE-LABEL: v4i16_select_ne
; CODE:       bb.0
; CODE-NEXT:    cmeq  v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    bit   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret

define <4 x i16> @v4i16_select_ne(<4 x i16> %a, <4 x i16> %b, <4 x i16> %c) {
  %cmp.1 = icmp ne <4 x i16> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x i16> %a, <4 x i16> %c
  ret <4 x i16> %s.1
}

; COST-LABEL: v8i16_select_ugt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = icmp ugt <8 x i16> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x i16> %a, <8 x i16> %c

; CODE-LABEL: v8i16_select_ugt
; CODE:       bb.0
; CODE-NEXT:    cmhi  v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

define <8 x i16> @v8i16_select_ugt(<8 x i16> %a, <8 x i16> %b, <8 x i16> %c) {
  %cmp.1 = icmp ugt <8 x i16> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x i16> %a, <8 x i16> %c
  ret <8 x i16> %s.1
}

; COST-LABEL: v2i32_select_ule
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = icmp ule <2 x i32> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x i32> %a, <2 x i32> %c

; CODE-LABEL: v2i32_select_ule
; CODE:       bb.0
; CODE-NEXT:    cmhs  v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret

define <2 x i32> @v2i32_select_ule(<2 x i32> %a, <2 x i32> %b, <2 x i32> %c) {
  %cmp.1 = icmp ule <2 x i32> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x i32> %a, <2 x i32> %c
  ret <2 x i32> %s.1
}

; COST-LABEL: v4i32_select_ult
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = icmp ult <4 x i32> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x i32> %a, <4 x i32> %c

; CODE-LABEL: v4i32_select_ult
; CODE:       bb.0
; CODE-NEXT:    cmhi  v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

define <4 x i32> @v4i32_select_ult(<4 x i32> %a, <4 x i32> %b, <4 x i32> %c) {
  %cmp.1 = icmp ult <4 x i32> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x i32> %a, <4 x i32> %c
  ret <4 x i32> %s.1
}

; COST-LABEL: v2i64_select_sle
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = icmp sle <2 x i64> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x i64> %a, <2 x i64> %c

; CODE-LABEL: v2i64_select_sle
; CODE:       bb.0
; CODE-NEXT:    cmge  v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

define <2 x i64> @v2i64_select_sle(<2 x i64> %a, <2 x i64> %b, <2 x i64> %c) {
  %cmp.1 = icmp sle <2 x i64> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x i64> %a, <2 x i64> %c
  ret <2 x i64> %s.1
}

; COST-LABEL: v3i64_select_sle
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %cmp.1 = icmp sle <3 x i64> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <3 x i1> %cmp.1, <3 x i64> %a, <3 x i64> %c

; CODE-LABEL: v3i64_select_sle
; CODE:       bb.0
; CODE:    mov
; CODE:    mov
; CODE:    mov
; CODE:    cmge
; CODE:    ldr
; CODE:    bif
; CODE:    cmge
; CODE:    bif
; CODE:    ext
; CODE:    ret

define <3 x i64> @v3i64_select_sle(<3 x i64> %a, <3 x i64> %b, <3 x i64> %c) {
  %cmp.1 = icmp sle <3 x i64> %a, %b
  %s.1 = select <3 x i1> %cmp.1, <3 x i64> %a, <3 x i64> %c
  ret <3 x i64> %s.1
}

; COST-LABEL: v2i64_select_no_cmp
; COST-NEXT:  Cost Model: Found an estimated cost of 6 for instruction:   %s.1 = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b

; CODE-LABEL: v2i64_select_no_cmp
; CODE:       bb.0
; CODE-NEXT:    ushll   v{{.+}}.2d, v{{.+}}.2s, #0
; CODE-NEXT:    shl v{{.+}}.2d, v{{.+}}.2d, #63
; CODE-NEXT:    cmlt    v{{.+}}.2d, v{{.+}}.2d, #0
; CODE-NEXT:    bif v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

define <2 x i64> @v2i64_select_no_cmp(<2 x i64> %a, <2 x i64> %b, <2 x i1> %cond) {
  %s.1 = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %s.1
}

define <4 x half> @v4f16_select_ogt(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; COST-LABEL: v4f16_select_ogt
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 4 for instruction:   %cmp.1 = fcmp ogt <4 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp ogt <4 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
;
; CODE-LABEL: v4f16_select_ogt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ogt <4 x half> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
  ret <4 x half> %s.1
}

define <8 x half> @v8f16_select_ogt(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; COST-LABEL: v8f16_select_ogt
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 22 for instruction:   %cmp.1 = fcmp ogt <8 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp ogt <8 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
;
; CODE-LABEL: v8f16_select_ogt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ogt <8 x half> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
  ret <8 x half> %s.1
}

define <2 x float> @v2f32_select_ogt(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_ogt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ogt <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
;
; CODE-LABEL: v2f32_select_ogt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ogt <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_ogt(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_ogt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ogt <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
;
; CODE-LABEL: v4f32_select_ogt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ogt <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_ogt(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_ogt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ogt <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_ogt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ogt <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}

define <4 x half> @v4f16_select_oge(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; COST-LABEL: v4f16_select_oge
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 4 for instruction:   %cmp.1 = fcmp oge <4 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp oge <4 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
;
; CODE-LABEL: v4f16_select_oge
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oge <4 x half> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
  ret <4 x half> %s.1
}

define <8 x half> @v8f16_select_oge(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; COST-LABEL: v8f16_select_oge
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 22 for instruction:   %cmp.1 = fcmp oge <8 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp oge <8 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
;
; CODE-LABEL: v8f16_select_oge
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oge <8 x half> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
  ret <8 x half> %s.1
}

define <2 x float> @v2f32_select_oge(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_oge
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp oge <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
;
; CODE-LABEL: v2f32_select_oge
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oge <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_oge(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_oge
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp oge <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
;
; CODE-LABEL: v4f32_select_oge
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oge <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_oge(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_oge
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp oge <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_oge
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oge <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}

define <4 x half> @v4f16_select_olt(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; COST-LABEL: v4f16_select_olt
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 4 for instruction:   %cmp.1 = fcmp olt <4 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp olt <4 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
;
; CODE-LABEL: v4f16_select_olt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp olt <4 x half> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
  ret <4 x half> %s.1
}

define <8 x half> @v8f16_select_olt(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; COST-LABEL: v8f16_select_olt
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 22 for instruction:   %cmp.1 = fcmp olt <8 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp olt <8 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
;
; CODE-LABEL: v8f16_select_olt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp olt <8 x half> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
  ret <8 x half> %s.1
}

define <2 x float> @v2f32_select_olt(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_olt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp olt <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
;
; CODE-LABEL: v2f32_select_olt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp olt <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_olt(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_olt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp olt <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
;
; CODE-LABEL: v4f32_select_olt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp olt <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_olt(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_olt
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp olt <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_olt
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp olt <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}

define <4 x half> @v4f16_select_ole(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; COST-LABEL: v4f16_select_ole
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 4 for instruction:   %cmp.1 = fcmp ole <4 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp ole <4 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
;
; CODE-LABEL: v4f16_select_ole
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ole <4 x half> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
  ret <4 x half> %s.1
}

define <8 x half> @v8f16_select_ole(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; COST-LABEL: v8f16_select_ole
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 22 for instruction:   %cmp.1 = fcmp ole <8 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp ole <8 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
;
; CODE-LABEL: v8f16_select_ole
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ole <8 x half> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
  ret <8 x half> %s.1
}

define <2 x float> @v2f32_select_ole(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_ole
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ole <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
;
; CODE-LABEL: v2f32_select_ole
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ole <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_ole(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_ole
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ole <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
;
; CODE-LABEL: v4f32_select_ole
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ole <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_ole(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_ole
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ole <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_ole
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ole <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}

define <4 x half> @v4f16_select_oeq(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; COST-LABEL: v4f16_select_oeq
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 4 for instruction:   %cmp.1 = fcmp oeq <4 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp oeq <4 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
;
; CODE-LABEL: v4f16_select_oeq
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oeq <4 x half> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
  ret <4 x half> %s.1
}

define <8 x half> @v8f16_select_oeq(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; COST-LABEL: v8f16_select_oeq
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 22 for instruction:   %cmp.1 = fcmp oeq <8 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp oeq <8 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
;
; CODE-LABEL: v8f16_select_oeq
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oeq <8 x half> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
  ret <8 x half> %s.1
}

define <2 x float> @v2f32_select_oeq(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_oeq
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp oeq <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
;
; CODE-LABEL: v2f32_select_oeq
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oeq <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_oeq(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_oeq
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp oeq <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
;
; CODE-LABEL: v4f32_select_oeq
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oeq <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_oeq(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_oeq
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp oeq <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_oeq
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp oeq <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}

define <4 x half> @v4f16_select_one(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; COST-LABEL: v4f16_select_one
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 4 for instruction:   %cmp.1 = fcmp one <4 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp one <4 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
;
; CODE-LABEL: v4f16_select_one
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    fcmgt v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    orr   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp one <4 x half> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
  ret <4 x half> %s.1
}

define <8 x half> @v8f16_select_one(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; COST-LABEL: v8f16_select_one
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 22 for instruction:  %cmp.1 = fcmp one <8 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
; COST-FULLFP16-NEXT: Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp one <8 x half> %a, %b
; COST-FULLFP16-NEXT: Cost Model: Found an estimated cost of 2 for instruction: %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
;
; CODE-LABEL: v8f16_select_one
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    fcmgt v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    orr   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp one <8 x half> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
  ret <8 x half> %s.1
}

define <2 x float> @v2f32_select_one(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_one
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp one <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c

; CODE-LABEL: v2f32_select_one
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    fcmgt v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    orr   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret

  %cmp.1 = fcmp one <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_one(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_one
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp one <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c

; CODE-LABEL: v4f32_select_one
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    fcmgt v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    orr   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

  %cmp.1 = fcmp one <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_one(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_one
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp one <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_one
; CODE:       bb.0
; CODE-NEXT:    fcmgt v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    fcmgt v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    orr   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp one <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}

define <4 x half> @v4f16_select_une(<4 x half> %a, <4 x half> %b, <4 x half> %c) {
; COST-LABEL: v4f16_select_une
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 4 for instruction:   %cmp.1 = fcmp une <4 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp une <4 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
;
; CODE-LABEL: v4f16_select_une
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.4h, v{{.+}}.4h, v{{.+}}.4h
; CODE-NEXT:    bit   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp une <4 x half> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x half> %a, <4 x half> %c
  ret <4 x half> %s.1
}

define <8 x half> @v8f16_select_une(<8 x half> %a, <8 x half> %b, <8 x half> %c) {
; COST-LABEL: v8f16_select_une
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 22 for instruction:   %cmp.1 = fcmp une <8 x half> %a, %b
; COST-NOFP16-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp une <8 x half> %a, %b
; COST-FULLFP16-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
;
; CODE-LABEL: v8f16_select_une
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.8h, v{{.+}}.8h, v{{.+}}.8h
; CODE-NEXT:    bit   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp une <8 x half> %a, %b
  %s.1 = select <8 x i1> %cmp.1, <8 x half> %a, <8 x half> %c
  ret <8 x half> %s.1
}

define <2 x float> @v2f32_select_une(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_une
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp une <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
;
; CODE-LABEL: v2f32_select_une
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    bit   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp une <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_une(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_une
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp une <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
;
; CODE-LABEL: v4f32_select_une
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    bit   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp une <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_une(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_une
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %cmp.1 = fcmp une <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_une
; CODE:       bb.0
; CODE-NEXT:    fcmeq v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    bit   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp une <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}

define <2 x float> @v2f32_select_ord(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
; COST-LABEL: v2f32_select_ord
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ord <2 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
;
; CODE-LABEL: v2f32_select_ord
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    fcmgt v{{.+}}.2s, v{{.+}}.2s, v{{.+}}.2s
; CODE-NEXT:    orr   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    bif   v{{.+}}.8b, v{{.+}}.8b, v{{.+}}.8b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ord <2 x float> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x float> %a, <2 x float> %c
  ret <2 x float> %s.1
}

define <4 x float> @v4f32_select_ord(<4 x float> %a, <4 x float> %b, <4 x float> %c) {
; COST-LABEL: v4f32_select_ord
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ord <4 x float> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c

; CODE-LABEL: v4f32_select_ord
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    fcmgt v{{.+}}.4s, v{{.+}}.4s, v{{.+}}.4s
; CODE-NEXT:    orr   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

  %cmp.1 = fcmp ord <4 x float> %a, %b
  %s.1 = select <4 x i1> %cmp.1, <4 x float> %a, <4 x float> %c
  ret <4 x float> %s.1
}

define <2 x double> @v2f64_select_ord(<2 x double> %a, <2 x double> %b, <2 x double> %c) {
; COST-LABEL: v2f64_select_ord
; COST-NEXT:  Cost Model: Found an estimated cost of 1 for instruction:   %cmp.1 = fcmp ord <2 x double> %a, %b
; COST-NEXT:  Cost Model: Found an estimated cost of 2 for instruction:   %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
;
; CODE-LABEL: v2f64_select_ord
; CODE:       bb.0
; CODE-NEXT:    fcmge v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    fcmgt v{{.+}}.2d, v{{.+}}.2d, v{{.+}}.2d
; CODE-NEXT:    orr   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    bif   v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret
;
  %cmp.1 = fcmp ord <2 x double> %a, %b
  %s.1 = select <2 x i1> %cmp.1, <2 x double> %a, <2 x double> %c
  ret <2 x double> %s.1
}
