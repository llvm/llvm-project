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
