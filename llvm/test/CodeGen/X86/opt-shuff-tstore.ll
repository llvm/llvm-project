; RUN: llc -mcpu=corei7 -mtriple=x86_64-linux < %s  -mattr=+sse2,+sse41 | FileCheck %s

; CHECK: func_4_8
; A single memory write
; CHECK: movd
; CHECK-NEXT: ret
define void @func_4_8(<4 x i8> %param, <4 x i8>* %p) {
  %r = add <4 x i8> %param, <i8 1, i8 2, i8 3, i8 4>
  store <4 x i8> %r, <4 x i8>* %p
  ret void
}

; CHECK: func_4_16
; CHECK: movq
; CHECK-NEXT: ret
define void @func_4_16(<4 x i16> %param, <4 x i16>* %p) {
  %r = add <4 x i16> %param, <i16 1, i16 2, i16 3, i16 4>
  store <4 x i16> %r, <4 x i16>* %p
  ret void
}

; CHECK: func_8_8
; CHECK: movq
; CHECK-NEXT: ret
define void @func_8_8(<8 x i8> %param, <8 x i8>* %p) {
  %r = add <8 x i8> %param, <i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4>
  store <8 x i8> %r, <8 x i8>* %p
  ret void
}

; CHECK: func_2_32
; CHECK: movq
; CHECK-NEXT: ret
define void @func_2_32(<2 x i32> %param, <2 x i32>* %p) {
  %r = add <2 x i32> %param, <i32 1, i32 2>
  store <2 x i32> %r, <2 x i32>* %p
  ret void
}

