; RUN: opt < %s -mtriple=aarch64--linux-gnu -cost-model -analyze | FileCheck %s --check-prefix=COST
; RUN: llc < %s -mtriple=aarch64--linux-gnu | FileCheck %s --check-prefix=CODE

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
; CODE:    ldr
; CODE:    mov
; CODE:    mov
; CODE:    cmge
; CODE:    cmge
; CODE:    bif
; CODE:    bif
; CODE:    ext
; CODE:    ret

define <3 x i64> @v3i64_select_sle(<3 x i64> %a, <3 x i64> %b, <3 x i64> %c) {
  %cmp.1 = icmp sle <3 x i64> %a, %b
  %s.1 = select <3 x i1> %cmp.1, <3 x i64> %a, <3 x i64> %c
  ret <3 x i64> %s.1
}

; COST-LABEL: v2i64_select_no_cmp
; COST-NEXT:  Cost Model: Found an estimated cost of 5 for instruction:   %s.1 = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b

; CODE-LABEL: v2i64_select_no_cmp
; CODE:       bb.0
; CODE-NEXT:    ushll   v{{.+}}.2d, v{{.+}}.2s, #0
; CODE-NEXT:    shl v{{.+}}.2d, v{{.+}}.2d, #63
; CODE-NEXT:    sshr    v{{.+}}.2d, v{{.+}}.2d, #63
; CODE-NEXT:    bif v{{.+}}.16b, v{{.+}}.16b, v{{.+}}.16b
; CODE-NEXT:    ret

define <2 x i64> @v2i64_select_no_cmp(<2 x i64> %a, <2 x i64> %b, <2 x i1> %cond) {
  %s.1 = select <2 x i1> %cond, <2 x i64> %a, <2 x i64> %b
  ret <2 x i64> %s.1
}
