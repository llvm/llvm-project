; RUN: llc < %s -mtriple=arm64-apple-ios7.0 -mcpu=cyclone | FileCheck %s

define <8 x i1> @test1() {
; CHECK-LABEL: test1:
; CHECK:       ; %bb.0: ; %entry
; CHECK-NEXT:    movi.16b v0, #0
; CHECK-NEXT:    ret
entry:
  %Shuff = shufflevector <8 x i1> <i1 0, i1 1, i1 2, i1 3, i1 4, i1 5, i1 6,
                                   i1 7>,
                         <8 x i1> <i1 0, i1 1, i1 2, i1 3, i1 4, i1 5, i1 6,
                                   i1 7>,
                         <8 x i32> <i32 2, i32 undef, i32 6, i32 undef, i32 10,
                                    i32 12, i32 14, i32 0>
  ret <8 x i1> %Shuff
}

define <8 x i1>@test2() {
; CHECK-LABEL: test2:
; CHECK:       ; %bb.0: ; %bb
; CHECK-NEXT:    movi d0, #0x0000ff00000000
; CHECK-NEXT:    ret
bb:
  %Shuff = shufflevector <8 x i1> zeroinitializer,
     <8 x i1> <i1 0, i1 1, i1 1, i1 0, i1 0, i1 1, i1 0, i1 0>,
     <8 x i32> <i32 2, i32 undef, i32 6, i32 undef, i32 10, i32 12, i32 14,
                i32 0>
  ret <8 x i1> %Shuff
}

define <16 x i1> @test3(i1* %ptr, i32 %v) {
; CHECK-LABEL: test3:
; CHECK:       ; %bb.0: ; %bb
; CHECK-NEXT:    movi.2d v0, #0x0000ff000000ff
; CHECK-NEXT:    ret
bb:
  %Shuff = shufflevector <16 x i1> <i1 0, i1 1, i1 1, i1 0, i1 0, i1 1, i1 0, i1 0, i1 0, i1 1, i1 1, i1 0, i1 0, i1 1, i1 0, i1 0>, <16 x i1> undef,
     <16 x i32> <i32 2, i32 undef, i32 6, i32 undef, i32 10, i32 12, i32 14,
                 i32 0, i32 2, i32 undef, i32 6, i32 undef, i32 10, i32 12,
                 i32 14, i32 0>
  ret <16 x i1> %Shuff
}
; CHECK-LABEL: lCPI3_0:
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   255                     ; 0xff
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
; CHECK:         .byte   0                       ; 0x0
define <16 x i1> @test4(i1* %ptr, i32 %v) {
; CHECK-LABEL: _test4:
; CHECK:         adrp    x[[REG3:[0-9]+]], lCPI3_0@PAGE
; CHECK:         ldr     q[[REG2:[0-9]+]], [x[[REG3]], lCPI3_0@PAGEOFF]
bb:
  %Shuff = shufflevector <16 x i1> zeroinitializer,
     <16 x i1> <i1 0, i1 1, i1 1, i1 0, i1 0, i1 1, i1 0, i1 0, i1 0, i1 1,
                i1 1, i1 0, i1 0, i1 1, i1 0, i1 0>,
     <16 x i32> <i32 2, i32 1, i32 6, i32 18, i32 10, i32 12, i32 14, i32 0,
                 i32 2, i32 31, i32 6, i32 30, i32 10, i32 12, i32 14, i32 0>
  ret <16 x i1> %Shuff
}
