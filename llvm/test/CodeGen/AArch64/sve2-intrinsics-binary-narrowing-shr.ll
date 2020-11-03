; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve2 < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

;
; SHRNB
;

define <vscale x 16 x i8> @shrnb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: shrnb_h:
; CHECK: shrnb z0.b, z0.h, #8
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.shrnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                 i32 8)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @shrnb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: shrnb_s:
; CHECK: shrnb z0.h, z0.s, #16
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.shrnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                 i32 16)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @shrnb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: shrnb_d:
; CHECK: shrnb z0.s, z0.d, #32
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.shrnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                 i32 32)
  ret <vscale x 4 x i32> %out
}

;
; UQSHRNB
;

define <vscale x 16 x i8> @uqshrnb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqshrnb_h:
; CHECK: uqshrnb z0.b, z0.h, #1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqshrnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                   i32 1)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqshrnb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqshrnb_s:
; CHECK: uqshrnb z0.h, z0.s, #1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqshrnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                   i32 1)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqshrnb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqshrnb_d:
; CHECK: uqshrnb z0.s, z0.d, #1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqshrnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                   i32 1)
  ret <vscale x 4 x i32> %out
}

;
; SQSHRNB
;

define <vscale x 16 x i8> @sqshrnb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqshrnb_h:
; CHECK: sqshrnb z0.b, z0.h, #1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqshrnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                   i32 1)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqshrnb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqshrnb_s:
; CHECK: sqshrnb z0.h, z0.s, #1
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqshrnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                   i32 1)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqshrnb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqshrnb_d:
; CHECK: sqshrnb z0.s, z0.d, #1
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshrnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                   i32 1)
  ret <vscale x 4 x i32> %out
}

;
; SQSHRUNB
;

define <vscale x 16 x i8> @sqshrunb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: qshrunb_h:
; CHECK: sqshrunb z0.b, z0.h, #7
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqshrunb.nxv8i16(<vscale x 8 x i16> %a,
                                                                    i32 7)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqshrunb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqshrunb_s:
; CHECK: sqshrunb z0.h, z0.s, #15
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqshrunb.nxv4i32(<vscale x 4 x i32> %a,
                                                                    i32 15)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqshrunb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqshrunb_d:
; CHECK: sqshrunb z0.s, z0.d, #31
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshrunb.nxv2i64(<vscale x 2 x i64> %a,
                                                                    i32 31)
  ret <vscale x 4 x i32> %out
}

;
; UQRSHRNB
;

define <vscale x 16 x i8> @uqrshrnb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: uqrshrnb_h:
; CHECK: uqrshrnb z0.b, z0.h, #2
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqrshrnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                    i32 2)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqrshrnb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: uqrshrnb_s:
; CHECK: uqrshrnb z0.h, z0.s, #2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqrshrnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                    i32 2)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqrshrnb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: uqrshrnb_d:
; CHECK: uqrshrnb z0.s, z0.d, #2
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqrshrnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                    i32 2)
  ret <vscale x 4 x i32> %out
}

;
; SQRSHRNB
;

define <vscale x 16 x i8> @sqrshrnb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqrshrnb_h:
; CHECK: sqrshrnb z0.b, z0.h, #2
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrnb.nxv8i16(<vscale x 8 x i16> %a,
                                                                    i32 2)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrshrnb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqrshrnb_s:
; CHECK: sqrshrnb z0.h, z0.s, #2
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrnb.nxv4i32(<vscale x 4 x i32> %a,
                                                                    i32 2)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrshrnb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqrshrnb_d:
; CHECK: sqrshrnb z0.s, z0.d, #2
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrnb.nxv2i64(<vscale x 2 x i64> %a,
                                                                    i32 2)
  ret <vscale x 4 x i32> %out
}

;
; SQRSHRUNB
;

define <vscale x 16 x i8> @sqrshrunb_h(<vscale x 8 x i16> %a) {
; CHECK-LABEL: sqrshrunb_h:
; CHECK: sqrshrunb z0.b, z0.h, #6
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrunb.nxv8i16(<vscale x 8 x i16> %a,
                                                                     i32 6)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrshrunb_s(<vscale x 4 x i32> %a) {
; CHECK-LABEL: sqrshrunb_s:
; CHECK: sqrshrunb z0.h, z0.s, #14
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrunb.nxv4i32(<vscale x 4 x i32> %a,
                                                                     i32 14)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrshrunb_d(<vscale x 2 x i64> %a) {
; CHECK-LABEL: sqrshrunb_d:
; CHECK: sqrshrunb z0.s, z0.d, #30
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrunb.nxv2i64(<vscale x 2 x i64> %a,
                                                                     i32 30)
  ret <vscale x 4 x i32> %out
}

;
; SHRNT
;

define <vscale x 16 x i8> @shrnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: shrnt_h:
; CHECK: shrnt z0.b, z1.h, #3
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.shrnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                 <vscale x 8 x i16> %b,
                                                                 i32 3)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @shrnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: shrnt_s:
; CHECK: shrnt z0.h, z1.s, #3
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.shrnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                 <vscale x 4 x i32> %b,
                                                                 i32 3)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @shrnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: shrnt_d:
; CHECK: shrnt z0.s, z1.d, #3
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.shrnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                 <vscale x 2 x i64> %b,
                                                                 i32 3)
  ret <vscale x 4 x i32> %out
}

;
; UQSHRNT
;

define <vscale x 16 x i8> @uqshrnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uqshrnt_h:
; CHECK: uqshrnt z0.b, z1.h, #5
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqshrnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                   <vscale x 8 x i16> %b,
                                                                   i32 5)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqshrnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uqshrnt_s:
; CHECK: uqshrnt z0.h, z1.s, #13
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqshrnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                   <vscale x 4 x i32> %b,
                                                                   i32 13)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqshrnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uqshrnt_d:
; CHECK: uqshrnt z0.s, z1.d, #29
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqshrnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                   <vscale x 2 x i64> %b,
                                                                   i32 29)
  ret <vscale x 4 x i32> %out
}

;
; SQSHRNT
;

define <vscale x 16 x i8> @sqshrnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqshrnt_h:
; CHECK: sqshrnt z0.b, z1.h, #5
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqshrnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                   <vscale x 8 x i16> %b,
                                                                   i32 5)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqshrnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqshrnt_s:
; CHECK: sqshrnt z0.h, z1.s, #13
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqshrnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                   <vscale x 4 x i32> %b,
                                                                   i32 13)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqshrnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqshrnt_d:
; CHECK: sqshrnt z0.s, z1.d, #29
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshrnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                   <vscale x 2 x i64> %b,
                                                                   i32 29)
  ret <vscale x 4 x i32> %out
}

;
; SQSHRUNT
;

define <vscale x 16 x i8> @sqshrunt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqshrunt_h:
; CHECK: sqshrunt z0.b, z1.h, #4
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqshrunt.nxv8i16(<vscale x 16 x i8> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    i32 4)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqshrunt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqshrunt_s:
; CHECK: sqshrunt z0.h, z1.s, #4
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqshrunt.nxv4i32(<vscale x 8 x i16> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    i32 4)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqshrunt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqshrunt_d:
; CHECK: sqshrunt z0.s, z1.d, #4
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqshrunt.nxv2i64(<vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    i32 4)
  ret <vscale x 4 x i32> %out
}

;
; UQRSHRNT
;

define <vscale x 16 x i8> @uqrshrnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: uqrshrnt_h:
; CHECK: uqrshrnt z0.b, z1.h, #8
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.uqrshrnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    i32 8)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @uqrshrnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: uqrshrnt_s:
; CHECK: uqrshrnt z0.h, z1.s, #12
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.uqrshrnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    i32 12)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @uqrshrnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: uqrshrnt_d:
; CHECK: uqrshrnt z0.s, z1.d, #28
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.uqrshrnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    i32 28)
  ret <vscale x 4 x i32> %out
}

;
; SQRSHRNT
;

define <vscale x 16 x i8> @sqrshrnt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqrshrnt_h:
; CHECK: sqrshrnt z0.b, z1.h, #8
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrnt.nxv8i16(<vscale x 16 x i8> %a,
                                                                    <vscale x 8 x i16> %b,
                                                                    i32 8)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrshrnt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqrshrnt_s:
; CHECK: sqrshrnt z0.h, z1.s, #12
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrnt.nxv4i32(<vscale x 8 x i16> %a,
                                                                    <vscale x 4 x i32> %b,
                                                                    i32 12)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrshrnt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqrshrnt_d:
; CHECK: sqrshrnt z0.s, z1.d, #28
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrnt.nxv2i64(<vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b,
                                                                    i32 28)
  ret <vscale x 4 x i32> %out
}

;
; SQRSHRUNT
;

define <vscale x 16 x i8> @sqrshrunt_h(<vscale x 16 x i8> %a, <vscale x 8 x i16> %b) {
; CHECK-LABEL: sqrshrunt_h:
; CHECK: sqrshrunt z0.b, z1.h, #1
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrunt.nxv8i16(<vscale x 16 x i8> %a,
                                                                     <vscale x 8 x i16> %b,
                                                                     i32 1)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @sqrshrunt_s(<vscale x 8 x i16> %a, <vscale x 4 x i32> %b) {
; CHECK-LABEL: sqrshrunt_s:
; CHECK: sqrshrunt z0.h, z1.s, #5
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrunt.nxv4i32(<vscale x 8 x i16> %a,
                                                                     <vscale x 4 x i32> %b,
                                                                     i32 5)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @sqrshrunt_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) {
; CHECK-LABEL: sqrshrunt_d:
; CHECK: sqrshrunt z0.s, z1.d, #5
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrunt.nxv2i64(<vscale x 4 x i32> %a,
                                                                     <vscale x 2 x i64> %b,
                                                                     i32 5)
  ret <vscale x 4 x i32> %out
}

declare <vscale x 16 x i8> @llvm.aarch64.sve.shrnb.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.shrnb.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.shrnb.nxv2i64(<vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqshrnb.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqshrnb.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqshrnb.nxv2i64(<vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqshrnb.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqshrnb.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqshrnb.nxv2i64(<vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqrshrnb.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqrshrnb.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqrshrnb.nxv2i64(<vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrnb.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrnb.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrnb.nxv2i64(<vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqshrunb.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqshrunb.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqshrunb.nxv2i64(<vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrunb.nxv8i16(<vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrunb.nxv4i32(<vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrunb.nxv2i64(<vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.shrnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.shrnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.shrnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqshrnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqshrnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqshrnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqshrnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqshrnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqshrnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqshrunt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqshrunt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqshrunt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.uqrshrnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.uqrshrnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.uqrshrnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrnt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrnt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrnt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, i32)

declare <vscale x 16 x i8> @llvm.aarch64.sve.sqrshrunt.nxv8i16(<vscale x 16 x i8>, <vscale x 8 x i16>, i32)
declare <vscale x 8 x i16> @llvm.aarch64.sve.sqrshrunt.nxv4i32(<vscale x 8 x i16>, <vscale x 4 x i32>, i32)
declare <vscale x 4 x i32> @llvm.aarch64.sve.sqrshrunt.nxv2i64(<vscale x 4 x i32>, <vscale x 2 x i64>, i32)
