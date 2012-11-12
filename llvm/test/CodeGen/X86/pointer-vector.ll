; RUN: llc < %s -mtriple=i686-linux -mcpu=corei7 | FileCheck %s
; RUN: opt -instsimplify %s -disable-output

;CHECK: SHUFF0
define <8 x i32*> @SHUFF0(<4 x i32*> %ptrv) nounwind {
entry:
  %G = shufflevector <4 x i32*> %ptrv, <4 x i32*> %ptrv, <8 x i32> <i32 2, i32 7, i32 1, i32 2, i32 4, i32 5, i32 1, i32 1>
;CHECK: pshufd
  ret <8 x i32*> %G
;CHECK: ret
}

;CHECK: SHUFF1
define <4 x i32*> @SHUFF1(<4 x i32*> %ptrv) nounwind {
entry:
  %G = shufflevector <4 x i32*> %ptrv, <4 x i32*> %ptrv, <4 x i32> <i32 2, i32 7, i32 7, i32 2>
;CHECK: pshufd
  ret <4 x i32*> %G
;CHECK: ret
}

;CHECK: SHUFF3
define <4 x i8*> @SHUFF3(<4 x i8*> %ptrv) nounwind {
entry:
  %G = shufflevector <4 x i8*> %ptrv, <4 x i8*> undef, <4 x i32> <i32 2, i32 7, i32 1, i32 2>
;CHECK: pshufd
  ret <4 x i8*> %G
;CHECK: ret
}

;CHECK: LOAD0
define <4 x i8*> @LOAD0(<4 x i8*>* %p) nounwind {
entry:
  %G = load <4 x i8*>* %p
;CHECK: movaps
  ret <4 x i8*> %G
;CHECK: ret
}

;CHECK: LOAD1
define <4 x i8*> @LOAD1(<4 x i8*>* %p) nounwind {
entry:
  %G = load <4 x i8*>* %p
;CHECK: movdqa
;CHECK: pshufd
;CHECK: movdqa
  %T = shufflevector <4 x i8*> %G, <4 x i8*> %G, <4 x i32> <i32 7, i32 1, i32 4, i32 3>
  store <4 x i8*> %T, <4 x i8*>* %p
  ret <4 x i8*> %G
;CHECK: ret
}

;CHECK: LOAD2
define <4 x i8*> @LOAD2(<4 x i8*>* %p) nounwind {
entry:
  %I = alloca <4 x i8*>
;CHECK: sub
  %G = load <4 x i8*>* %p
;CHECK: movaps
  store <4 x i8*> %G, <4 x i8*>* %I
;CHECK: movaps
  %Z = load <4 x i8*>* %I
  ret <4 x i8*> %Z
;CHECK: add
;CHECK: ret
}

;CHECK: INT2PTR0
define <4 x i32> @INT2PTR0(<4 x i8*>* %p) nounwind {
entry:
  %G = load <4 x i8*>* %p
;CHECK: movl
;CHECK: movaps
  %K = ptrtoint <4 x i8*> %G to <4 x i32>
;CHECK: ret
  ret <4 x i32> %K
}

;CHECK: INT2PTR1
define <4 x i32*> @INT2PTR1(<4 x i8>* %p) nounwind {
entry:
  %G = load <4 x i8>* %p
;CHECK: movl
;CHECK: pmovzxbd
;CHECK: pand
  %K = inttoptr <4 x i8> %G to <4 x i32*>
;CHECK: ret
  ret <4 x i32*> %K
}

;CHECK: BITCAST0
define <4 x i32*> @BITCAST0(<4 x i8*>* %p) nounwind {
entry:
  %G = load <4 x i8*>* %p
;CHECK: movl
  %T = bitcast <4 x i8*> %G to <4 x i32*>
;CHECK: movaps
;CHECK: ret
  ret <4 x i32*> %T
}

;CHECK: BITCAST1
define <2 x i32*> @BITCAST1(<2 x i8*>* %p) nounwind {
entry:
  %G = load <2 x i8*>* %p
;CHECK: movl
;CHECK: pmovzxdq
  %T = bitcast <2 x i8*> %G to <2 x i32*>
;CHECK: ret
  ret <2 x i32*> %T
}

;CHECK: ICMP0
define <4 x i32> @ICMP0(<4 x i8*>* %p0, <4 x i8*>* %p1) nounwind {
entry:
  %g0 = load <4 x i8*>* %p0
  %g1 = load <4 x i8*>* %p1
  %k = icmp sgt <4 x i8*> %g0, %g1
  ;CHECK: pcmpgtd
  %j = select <4 x i1> %k, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, <4 x i32> <i32 9, i32 8, i32 7, i32 6>
  ret <4 x i32> %j
  ;CHECK: ret
}

;CHECK: ICMP1
define <4 x i32> @ICMP1(<4 x i8*>* %p0, <4 x i8*>* %p1) nounwind {
entry:
  %g0 = load <4 x i8*>* %p0
  %g1 = load <4 x i8*>* %p1
  %k = icmp eq <4 x i8*> %g0, %g1
  ;CHECK: pcmpeqd
  %j = select <4 x i1> %k, <4 x i32> <i32 0, i32 1, i32 2, i32 4>, <4 x i32> <i32 9, i32 8, i32 7, i32 6>
  ret <4 x i32> %j
  ;CHECK: ret
}

