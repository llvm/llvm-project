; Bytes and halves of a global's address. The address is an i32 load from
; the literal pool, which DAGCombine must not narrow to an i8/i16 load:
; Xtensa can only read a literal with L32R, and the address of a literal
; (XtensaISD::PCREL_WRAPPER) is not selectable on its own.
; Correct code has no l8ui/l16ui/l16si, only L32R and register operations.
;
; RUN: llc -mtriple=xtensa -O2 < %s | FileCheck %s
; RUN: llc -mtriple=xtensa -mcpu=esp32s3 -O2 < %s | FileCheck %s
; RUN: llc -mtriple=xtensa -mcpu=esp32s3 -O0 < %s | FileCheck %s

@g = external global [8 x i8]

; (unsigned)&g & 0xff
define i32 @low_byte() {
; CHECK-LABEL: low_byte:
; CHECK:       l32r
; CHECK-NOT:   {{l8ui|l16ui|l16si}}
; CHECK:       ret
  %p = ptrtoint ptr @g to i32
  %r = and i32 %p, 255
  ret i32 %r
}

; ((unsigned)&g >> 8) & 0xff
define i32 @second_byte() {
; CHECK-LABEL: second_byte:
; CHECK:       l32r
; CHECK-NOT:   {{l8ui|l16ui|l16si}}
; CHECK:       ret
  %p = ptrtoint ptr @g to i32
  %s = lshr i32 %p, 8
  %r = and i32 %s, 255
  ret i32 %r
}

; (unsigned)&g >> 24
define i32 @top_byte() {
; CHECK-LABEL: top_byte:
; CHECK:       l32r
; CHECK-NOT:   {{l8ui|l16ui|l16si}}
; CHECK:       ret
  %p = ptrtoint ptr @g to i32
  %r = lshr i32 %p, 24
  ret i32 %r
}

; (unsigned)&g >> 16
define i32 @high_half() {
; CHECK-LABEL: high_half:
; CHECK:       l32r
; CHECK-NOT:   {{l8ui|l16ui|l16si}}
; CHECK:       ret
  %p = ptrtoint ptr @g to i32
  %r = lshr i32 %p, 16
  ret i32 %r
}

; (int)&g >> 16
define i32 @high_half_signed() {
; CHECK-LABEL: high_half_signed:
; CHECK:       l32r
; CHECK-NOT:   {{l8ui|l16ui|l16si}}
; CHECK:       ret
  %p = ptrtoint ptr @g to i32
  %r = ashr i32 %p, 16
  ret i32 %r
}

; ((unsigned)&g >> 24) == 0x3f, a memory region check
define i1 @in_region() {
; CHECK-LABEL: in_region:
; CHECK:       l32r
; CHECK-NOT:   {{l8ui|l16ui|l16si}}
; CHECK:       ret
  %p = ptrtoint ptr @g to i32
  %s = lshr i32 %p, 24
  %r = icmp eq i32 %s, 63
  ret i1 %r
}
