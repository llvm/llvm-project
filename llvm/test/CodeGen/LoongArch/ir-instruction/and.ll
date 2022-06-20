; RUN: llc --mtriple=loongarch32 < %s | FileCheck %s --check-prefix=LA32
; RUN: llc --mtriple=loongarch64 < %s | FileCheck %s --check-prefix=LA64

;; Exercise the 'and' LLVM IR: https://llvm.org/docs/LangRef.html#and-instruction

define i1 @and_i1(i1 %a, i1 %b) {
; LA32-LABEL: and_i1:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i1:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i1 %a, %b
  ret i1 %r
}

define i8 @and_i8(i8 %a, i8 %b) {
; LA32-LABEL: and_i8:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i8:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i8 %a, %b
  ret i8 %r
}

define i16 @and_i16(i16 %a, i16 %b) {
; LA32-LABEL: and_i16:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i16:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i16 %a, %b
  ret i16 %r
}

define i32 @and_i32(i32 %a, i32 %b) {
; LA32-LABEL: and_i32:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i32:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i32 %a, %b
  ret i32 %r
}

define i64 @and_i64(i64 %a, i64 %b) {
; LA32-LABEL: and_i64:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    and $a0, $a0, $a2
; LA32-NEXT:    and $a1, $a1, $a3
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i64:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i64 %a, %b
  ret i64 %r
}

define i1 @and_i1_0(i1 %b) {
; LA32-LABEL: and_i1_0:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    move $a0, $zero
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i1_0:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    move $a0, $zero
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i1 4, %b
  ret i1 %r
}

define i1 @and_i1_5(i1 %b) {
; LA32-LABEL: and_i1_5:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i1_5:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i1 5, %b
  ret i1 %r
}

define i8 @and_i8_5(i8 %b) {
; LA32-LABEL: and_i8_5:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    andi $a0, $a0, 5
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i8_5:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    andi $a0, $a0, 5
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i8 5, %b
  ret i8 %r
}

define i8 @and_i8_257(i8 %b) {
; LA32-LABEL: and_i8_257:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i8_257:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i8 257, %b
  ret i8 %r
}

define i16 @and_i16_5(i16 %b) {
; LA32-LABEL: and_i16_5:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    andi $a0, $a0, 5
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i16_5:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    andi $a0, $a0, 5
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i16 5, %b
  ret i16 %r
}

define i16 @and_i16_0x1000(i16 %b) {
; LA32-LABEL: and_i16_0x1000:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    lu12i.w $a1, 1
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i16_0x1000:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    lu12i.w $a1, 1
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i16 4096, %b
  ret i16 %r
}

define i16 @and_i16_0x10001(i16 %b) {
; LA32-LABEL: and_i16_0x10001:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i16_0x10001:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i16 65537, %b
  ret i16 %r
}

define i32 @and_i32_5(i32 %b) {
; LA32-LABEL: and_i32_5:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    andi $a0, $a0, 5
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i32_5:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    andi $a0, $a0, 5
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i32 5, %b
  ret i32 %r
}

define i32 @and_i32_0x1000(i32 %b) {
; LA32-LABEL: and_i32_0x1000:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    lu12i.w $a1, 1
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i32_0x1000:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    lu12i.w $a1, 1
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i32 4096, %b
  ret i32 %r
}

define i32 @and_i32_0x100000001(i32 %b) {
; LA32-LABEL: and_i32_0x100000001:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    andi $a0, $a0, 1
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i32_0x100000001:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    andi $a0, $a0, 1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i32 4294967297, %b
  ret i32 %r
}

define i64 @and_i64_5(i64 %b) {
; LA32-LABEL: and_i64_5:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    andi $a0, $a0, 5
; LA32-NEXT:    move $a1, $zero
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i64_5:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    andi $a0, $a0, 5
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i64 5, %b
  ret i64 %r
}

define i64 @and_i64_0x1000(i64 %b) {
; LA32-LABEL: and_i64_0x1000:
; LA32:       # %bb.0: # %entry
; LA32-NEXT:    lu12i.w $a1, 1
; LA32-NEXT:    and $a0, $a0, $a1
; LA32-NEXT:    move $a1, $zero
; LA32-NEXT:    jirl $zero, $ra, 0
;
; LA64-LABEL: and_i64_0x1000:
; LA64:       # %bb.0: # %entry
; LA64-NEXT:    lu12i.w $a1, 1
; LA64-NEXT:    and $a0, $a0, $a1
; LA64-NEXT:    jirl $zero, $ra, 0
entry:
  %r = and i64 4096, %b
  ret i64 %r
}
