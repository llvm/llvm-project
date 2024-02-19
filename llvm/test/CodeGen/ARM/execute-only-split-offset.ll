; RUN: llc -mtriple=thumbv8m.base-eabi -mattr=+execute-only %s -o - | FileCheck --check-prefixes=CHECK,CHECK-MOVW %s
; RUN: llc -mtriple=thumbv6m-eabi -mattr=+execute-only %s -o - | FileCheck --check-prefixes=CHECK,CHECK-NOMOVW %s

; Largest offset that fits into sp-relative ldr
; CHECK-LABEL: ldr_range_end:
; CHECK: ldr {{r[0-9]+}}, [sp, #1020]
define i32 @ldr_range_end() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [1020 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; Smallest offset that fits into add+ldr
; CHECK-LABEL: add_ldr_range_start:
; CHECK:      add [[REG:r[0-9]+]], sp, #900
; CHECK-NEXT: ldr {{r[0-9]+}}, [[[REG]], #124]
define i32 @add_ldr_range_start() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [1024 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; Largest offset that fits into add+ldr
; CHECK-LABEL: add_ldr_range_end:
; CHECK:      add [[REG:r[0-9]+]], sp, #1020
; CHECK-NEXT: ldr {{r[0-9]+}}, [[[REG]], #124]
define i32 @add_ldr_range_end() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [1144 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; Smallest offset where we start using mov32. If we don't have movw then using
; an ldr offset means we save an add.
; CHECK-LABEL: mov32_range_start:
; CHECK-MOVW:        movw [[REG:r[0-9]+]], #1148
; CHECK-NOMOVW:      movs [[REG:r[0-9]+]], #4
; CHECK-NOMOVW-NEXT: lsls [[REG]], [[REG]], #8
; CHECK-NEXT:        add [[REG]], sp
; CHECK-MOVW-NEXT:   ldr {{r[0-9]+}}, [[[REG]]]
; CHECK-NOMOVW-NEXT: ldr {{r[0-9]+}}, [[[REG]], #124]
define i32 @mov32_range_start() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [1148 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; Here using an ldr offset doesn't save an add so we shouldn't do it.
; CHECK-LABEL: mov32_range_next:
; CHECK-MOVW:        movw [[REG:r[0-9]+]], #1152
; CHECK-NOMOVW:      movs [[REG:r[0-9]+]], #4
; CHECK-NOMOVW-NEXT: lsls [[REG]], [[REG]], #8
; CHECK-NOMOVW-NEXT: adds [[REG]], #128
; CHECK-NEXT:        add [[REG]], sp
; CHECK-NEXT:        ldr {{r[0-9]+}}, [[[REG]]]
define i32 @mov32_range_next() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [1152 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; Smallest offset where using an ldr offset prevents needing a movt or lsl+add
; CHECK-LABEL: can_clear_top_byte_start:
; CHECK:             add sp, {{r[0-9]+}}
; CHECK-MOVW:        movw [[REG:r[0-9]+]], #65412
; CHECK-NOMOVW:      movs [[REG:r[0-9]+]], #255
; CHECK-NOMOVW-NEXT: lsls [[REG:r[0-9]+]], [[REG:r[0-9]+]], #8
; CHECK-NOMOVW-NEXT: adds [[REG:r[0-9]+]], #132
; CHECK-NEXT:        add [[REG]], sp
; CHECK-NEXT:        ldr {{r[0-9]+}}, [[[REG]], #124]
define i32 @can_clear_top_byte_start() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [65536 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; Largest offset where using an ldr offset prevents needing a movt or lsl+add
; CHECK-LABEL: can_clear_top_byte_end:
; CHECK:             add sp, {{r[0-9]+}}
; CHECK-MOVW:        movw [[REG:r[0-9]+]], #65532
; CHECK-NOMOVW:      movs [[REG:r[0-9]+]], #255
; CHECK-NOMOVW-NEXT: lsls [[REG:r[0-9]+]], [[REG:r[0-9]+]], #8
; CHECK-NOMOVW-NEXT: adds [[REG:r[0-9]+]], #252
; CHECK-NEXT:        add [[REG]], sp
; CHECK-NEXT:        ldr {{r[0-9]+}}, [[[REG]], #124]
define i32 @can_clear_top_byte_end() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [65656 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; Smallest offset where using an ldr offset doesn't clear the top byte, though
; we can use an ldr offset if not using movt to save an add of the low byte.
; CHECK-LABEL: cant_clear_top_byte_start:
; CHECK:             add sp, {{r[0-9]+}}
; CHECK-MOVW:        movw [[REG:r[0-9]+]], #124
; CHECK-MOVW-NEXT:   movt [[REG:r[0-9]+]], #1
; CHECK-NOMOVW:      movs [[REG:r[0-9]+]], #1
; CHECK-NOMOVW-NEXT: lsls [[REG:r[0-9]+]], [[REG:r[0-9]+]], #16
; CHECK-NEXT:        add [[REG]], sp
; CHECK-MOVW-NEXT:   ldr {{r[0-9]+}}, [[[REG]]]
; CHECK-NOMOVW-NEXT: ldr {{r[0-9]+}}, [[[REG]], #124]
define i32 @cant_clear_top_byte_start() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [65660 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}

; An ldr offset doesn't help for anything, so we shouldn't do it.
; CHECK-LABEL: cant_clear_top_byte_next:
; CHECK:             add sp, {{r[0-9]+}}
; CHECK-MOVW:        movw [[REG:r[0-9]+]], #128
; CHECK-MOVW:        movt [[REG:r[0-9]+]], #1
; CHECK-NOMOVW:      movs [[REG:r[0-9]+]], #1
; CHECK-NOMOVW-NEXT: lsls [[REG:r[0-9]+]], [[REG:r[0-9]+]], #16
; CHECK-NOMOVW-NEXT: adds [[REG:r[0-9]+]], #128
; CHECK-NEXT:        add [[REG]], sp
; CHECK-NEXT:        ldr {{r[0-9]+}}, [[[REG]]]
define i32 @cant_clear_top_byte_next() {
entry:
  %var = alloca i32, align 4
  %arr = alloca [65664 x i8], align 4
  %0 = load i32, ptr %var, align 4
  ret i32 %0
}
