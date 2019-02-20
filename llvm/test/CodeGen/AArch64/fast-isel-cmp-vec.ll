; RUN: llc -mtriple=aarch64-apple-darwin -fast-isel -verify-machineinstrs \
; RUN:   -aarch64-enable-atomic-cfg-tidy=0 -disable-cgp -disable-branch-fold \
; RUN:   < %s | FileCheck %s

;
; Verify that we don't mess up vector comparisons in fast-isel.
;

define <2 x i32> @icmp_v2i32(<2 x i32> %a) {
; CHECK-LABEL: icmp_v2i32:
; CHECK:      ; %bb.0:
; CHECK-NEXT:  cmeq.2s [[CMP:v[0-9]+]], v0, #0
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  movi.2s [[MASK:v[0-9]+]], #1
; CHECK-NEXT:  and.8b v0, [[CMP]], [[MASK]]
; CHECK-NEXT:  ret
  %c = icmp eq <2 x i32> %a, zeroinitializer
  br label %bb2
bb2:
  %z = zext <2 x i1> %c to <2 x i32>
  ret <2 x i32> %z
}

define <2 x i32> @icmp_constfold_v2i32(<2 x i32> %a) {
; CHECK-LABEL: icmp_constfold_v2i32:
; CHECK:      ; %bb.0:
; CHECK-NEXT:  movi.2d v[[CMP:[0-9]+]], #0xffffffffffffffff
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  movi.2s [[MASK:v[0-9]+]], #1
; CHECK-NEXT:  and.8b v0, v[[CMP]], [[MASK]]
; CHECK-NEXT:  ret
  %1 = icmp eq <2 x i32> %a, %a
  br label %bb2
bb2:
  %2 = zext <2 x i1> %1 to <2 x i32>
  ret <2 x i32> %2
}

define <4 x i32> @icmp_v4i32(<4 x i32> %a) {
; CHECK-LABEL: icmp_v4i32:
; CHECK:      ; %bb.0:
; CHECK-NEXT:  cmeq.4s [[CMP:v[0-9]+]], v0, #0
; CHECK-NEXT:  xtn.4h [[CMPV4I16:v[0-9]+]], [[CMP]]
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  movi.4h [[MASK:v[0-9]+]], #1
; CHECK-NEXT:  and.8b [[ZEXT:v[0-9]+]], [[CMPV4I16]], [[MASK]]
; CHECK-NEXT:  ushll.4s v0, [[ZEXT]], #0
; CHECK-NEXT:  ret
  %c = icmp eq <4 x i32> %a, zeroinitializer
  br label %bb2
bb2:
  %z = zext <4 x i1> %c to <4 x i32>
  ret <4 x i32> %z
}

define <4 x i32> @icmp_constfold_v4i32(<4 x i32> %a) {
; CHECK-LABEL: icmp_constfold_v4i32:
; CHECK:      ; %bb.0:
; CHECK-NEXT:  movi.2d v[[CMP:[0-9]+]], #0xffffffffffffffff
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  movi.4h [[MASK:v[0-9]+]], #1
; CHECK-NEXT:  and.8b [[ZEXT:v[0-9]+]], v[[CMP]], [[MASK]]
; CHECK-NEXT:  ushll.4s v0, [[ZEXT]], #0
; CHECK-NEXT:  ret
  %1 = icmp eq <4 x i32> %a, %a
  br label %bb2
bb2:
  %2 = zext <4 x i1> %1 to <4 x i32>
  ret <4 x i32> %2
}

define <16 x i8> @icmp_v16i8(<16 x i8> %a) {
; CHECK-LABEL: icmp_v16i8:
; CHECK:      ; %bb.0:
; CHECK-NEXT:  cmeq.16b [[CMP:v[0-9]+]], v0, #0
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  movi.16b [[MASK:v[0-9]+]], #1
; CHECK-NEXT:  and.16b v0, [[CMP]], [[MASK]]
; CHECK-NEXT:  ret
  %c = icmp eq <16 x i8> %a, zeroinitializer
  br label %bb2
bb2:
  %z = zext <16 x i1> %c to <16 x i8>
  ret <16 x i8> %z
}

define <16 x i8> @icmp_constfold_v16i8(<16 x i8> %a) {
; CHECK-LABEL: icmp_constfold_v16i8:
; CHECK:      ; %bb.0:
; CHECK-NEXT:  movi.2d [[CMP:v[0-9]+]], #0xffffffffffffffff
; CHECK-NEXT: ; %bb.1:
; CHECK-NEXT:  movi.16b [[MASK:v[0-9]+]], #1
; CHECK-NEXT:  and.16b v0, [[CMP]], [[MASK]]
; CHECK-NEXT:  ret
  %1 = icmp eq <16 x i8> %a, %a
  br label %bb2
bb2:
  %2 = zext <16 x i1> %1 to <16 x i8>
  ret <16 x i8> %2
}
