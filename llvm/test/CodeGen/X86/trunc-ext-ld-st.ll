; RUN: llc < %s -march=x86-64 -mcpu=corei7 -mattr=+sse41 | FileCheck %s

;CHECK: load_2_i8
; A single 16-bit load
;CHECK: pmovzxbq
;CHECK: paddq
;CHECK: pshufb
; A single 16-bit store
;CHECK: movw
;CHECK: ret

define void @load_2_i8(<2 x i8>* %A)  {
   %T = load <2 x i8>* %A
   %G = add <2 x i8> %T, <i8 9, i8 7>
   store <2 x i8> %G, <2 x i8>* %A
   ret void
} 

;CHECK: load_2_i16
; Read 32-bits
;CHECK: pmovzxwq
;CHECK: paddq
;CHECK: pshufb
;CHECK: movd
;CHECK: ret
define void @load_2_i16(<2 x i16>* %A)  {
   %T = load <2 x i16>* %A
   %G = add <2 x i16> %T, <i16 9, i16 7>
   store <2 x i16> %G, <2 x i16>* %A
   ret void
} 

;CHECK: load_2_i32
;CHECK: pmovzxdq
;CHECK: paddq
;CHECK: pshufd
;CHECK: ret
define void @load_2_i32(<2 x i32>* %A)  {
   %T = load <2 x i32>* %A
   %G = add <2 x i32> %T, <i32 9, i32 7>
   store <2 x i32> %G, <2 x i32>* %A
   ret void
} 

;CHECK: load_4_i8
;CHECK: pmovzxbd
;CHECK: paddd
;CHECK: pshufb
;CHECK: ret
define void @load_4_i8(<4 x i8>* %A)  {
   %T = load <4 x i8>* %A
   %G = add <4 x i8> %T, <i8 1, i8 4, i8 9, i8 7>
   store <4 x i8> %G, <4 x i8>* %A
   ret void
} 

;CHECK: load_4_i16
;CHECK: pmovzxwd
;CHECK: paddd
;CHECK: pshufb
;CHECK: ret
define void @load_4_i16(<4 x i16>* %A)  {
   %T = load <4 x i16>* %A
   %G = add <4 x i16> %T, <i16 1, i16 4, i16 9, i16 7>
   store <4 x i16> %G, <4 x i16>* %A
   ret void
} 

;CHECK: load_8_i8
;CHECK: pmovzxbw
;CHECK: paddw
;CHECK: pshufb
;CHECK: ret
define void @load_8_i8(<8 x i8>* %A)  {
   %T = load <8 x i8>* %A
   %G = add <8 x i8> %T, %T
   store <8 x i8> %G, <8 x i8>* %A
   ret void
} 
