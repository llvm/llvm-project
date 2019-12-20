; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s | FileCheck %s

;
; CNTB
;

define i64 @cntb() {
; CHECK-LABEL: cntb:
; CHECK: cntb x0, vl2
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cntb(i32 2)
  ret i64 %out
}

define i64 @cntb_mul3() {
; CHECK-LABEL: cntb_mul3:
; CHECK: cntb x0, vl6, mul #3
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cntb(i32 6)
  %out = mul i64 %cnt, 3
  ret i64 %out
}

define i64 @cntb_mul4() {
; CHECK-LABEL: cntb_mul4:
; CHECK: cntb x0, vl8, mul #4
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cntb(i32 8)
  %out = mul i64 %cnt, 4
  ret i64 %out
}

;
; CNTH
;

define i64 @cnth() {
; CHECK-LABEL: cnth:
; CHECK: cnth x0, vl3
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cnth(i32 3)
  ret i64 %out
}

define i64 @cnth_mul5() {
; CHECK-LABEL: cnth_mul5:
; CHECK: cnth x0, vl7, mul #5
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cnth(i32 7)
  %out = mul i64 %cnt, 5
  ret i64 %out
}

define i64 @cnth_mul8() {
; CHECK-LABEL: cnth_mul8:
; CHECK: cnth x0, vl5, mul #8
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cnth(i32 5)
  %out = mul i64 %cnt, 8
  ret i64 %out
}

;
; CNTW
;

define i64 @cntw() {
; CHECK-LABEL: cntw:
; CHECK: cntw x0, vl4
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cntw(i32 4)
  ret i64 %out
}

define i64 @cntw_mul11() {
; CHECK-LABEL: cntw_mul11:
; CHECK: cntw x0, vl8, mul #11
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cntw(i32 8)
  %out = mul i64 %cnt, 11
  ret i64 %out
}

define i64 @cntw_mul2() {
; CHECK-LABEL: cntw_mul2:
; CHECK: cntw x0, vl6, mul #2
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cntw(i32 6)
  %out = mul i64 %cnt, 2
  ret i64 %out
}

;
; CNTD
;

define i64 @cntd() {
; CHECK-LABEL: cntd:
; CHECK: cntd x0, vl5
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cntd(i32 5)
  ret i64 %out
}

define i64 @cntd_mul15() {
; CHECK-LABEL: cntd_mul15:
; CHECK: cntd x0, vl16, mul #15
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cntd(i32 9)
  %out = mul i64 %cnt, 15
  ret i64 %out
}

define i64 @cntd_mul16() {
; CHECK-LABEL: cntd_mul16:
; CHECK: cntd x0, vl32, mul #16
; CHECK-NEXT: ret
  %cnt = call i64 @llvm.aarch64.sve.cntd(i32 10)
  %out = mul i64 %cnt, 16
  ret i64 %out
}

;
; CNTP
;

define i64 @cntp_b8(<vscale x 16 x i1> %pg, <vscale x 16 x i1> %a) {
; CHECK-LABEL: cntp_b8:
; CHECK: cntp x0, p0, p1.b
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cntp.nxv16i1(<vscale x 16 x i1> %pg,
                                                 <vscale x 16 x i1> %a)
  ret i64 %out
}

define i64 @cntp_b16(<vscale x 8 x i1> %pg, <vscale x 8 x i1> %a) {
; CHECK-LABEL: cntp_b16:
; CHECK: cntp x0, p0, p1.h
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cntp.nxv8i1(<vscale x 8 x i1> %pg,
                                                <vscale x 8 x i1> %a)
  ret i64 %out
}

define i64 @cntp_b32(<vscale x 4 x i1> %pg, <vscale x 4 x i1> %a) {
; CHECK-LABEL: cntp_b32:
; CHECK: cntp x0, p0, p1.s
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cntp.nxv4i1(<vscale x 4 x i1> %pg,
                                                <vscale x 4 x i1> %a)
  ret i64 %out
}

define i64 @cntp_b64(<vscale x 2 x i1> %pg, <vscale x 2 x i1> %a) {
; CHECK-LABEL: cntp_b64:
; CHECK: cntp x0, p0, p1.d
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.cntp.nxv2i1(<vscale x 2 x i1> %pg,
                                                <vscale x 2 x i1> %a)
  ret i64 %out
}

declare i64 @llvm.aarch64.sve.cntb(i32 %pattern)
declare i64 @llvm.aarch64.sve.cnth(i32 %pattern)
declare i64 @llvm.aarch64.sve.cntw(i32 %pattern)
declare i64 @llvm.aarch64.sve.cntd(i32 %pattern)

declare i64 @llvm.aarch64.sve.cntp.nxv16i1(<vscale x 16 x i1>, <vscale x 16 x i1>)
declare i64 @llvm.aarch64.sve.cntp.nxv8i1(<vscale x 8 x i1>, <vscale x 8 x i1>)
declare i64 @llvm.aarch64.sve.cntp.nxv4i1(<vscale x 4 x i1>, <vscale x 4 x i1>)
declare i64 @llvm.aarch64.sve.cntp.nxv2i1(<vscale x 2 x i1>, <vscale x 2 x i1>)
