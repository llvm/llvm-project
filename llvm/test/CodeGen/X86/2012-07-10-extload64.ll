; RUN: llc < %s -march=x86 -mcpu=corei7 -mtriple=i686-pc-win32 | FileCheck %s

; CHECK: load_store
define void @load_store(<4 x i16>* %in) {
entry:
; CHECK: pmovzxwd
  %A27 = load <4 x i16>* %in, align 4
  %A28 = add <4 x i16> %A27, %A27
; CHECK: movlpd
  store <4 x i16> %A28, <4 x i16>* %in, align 4
  ret void
; CHECK: ret
}

; Make sure that we store a 64bit value, even on 32bit systems.
;CHECK: store_64
define void @store_64(<2 x i32>* %ptr) {
BB:
  store <2 x i32> zeroinitializer, <2 x i32>* %ptr
  ret void
;CHECK: movlpd
;CHECK: ret
}

;CHECK: load_64
define <2 x i32> @load_64(<2 x i32>* %ptr) {
BB:
  %t = load <2 x i32>* %ptr
  ret <2 x i32> %t
;CHECK: pmovzxdq
;CHECK: ret
}
