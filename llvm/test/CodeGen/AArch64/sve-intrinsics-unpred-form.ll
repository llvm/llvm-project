; RUN: llc < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

;
; MUL
;

define <vscale x 16 x i8> @mul_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) #0 {
; CHECK-LABEL: mul_i8:
; CHECK: mul z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.mul.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @mul_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) #0 {
; CHECK-LABEL: mul_i16:
; CHECK: mul z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @mul_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: mul_i32:
; CHECK: mul z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.mul.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @mul_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: mul_i64:
; CHECK: mul z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; SMULH
;

define <vscale x 16 x i8> @smulh_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) #0 {
; CHECK-LABEL: smulh_i8:
; CHECK: smulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.smulh.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @smulh_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) #0 {
; CHECK-LABEL: smulh_i16:
; CHECK: smulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.smulh.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @smulh_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: smulh_i32:
; CHECK: smulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.smulh.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @smulh_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: smulh_i64:
; CHECK: smulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.smulh.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

;
; UMULH
;

define <vscale x 16 x i8> @umulh_i8(<vscale x 16 x i8> %a, <vscale x 16 x i8> %b) #0 {
; CHECK-LABEL: umulh_i8:
; CHECK: umulh z0.b, z0.b, z1.b
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.umulh.nxv16i8(<vscale x 16 x i1> %pg,
                                                                 <vscale x 16 x i8> %a,
                                                                 <vscale x 16 x i8> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @umulh_i16(<vscale x 8 x i16> %a, <vscale x 8 x i16> %b) #0 {
; CHECK-LABEL: umulh_i16:
; CHECK: umulh z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.umulh.nxv8i16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x i16> %a,
                                                                 <vscale x 8 x i16> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @umulh_i32(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: umulh_i32:
; CHECK: umulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.umulh.nxv4i32(<vscale x 4 x i1> %pg,
                                                                 <vscale x 4 x i32> %a,
                                                                 <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @umulh_i64(<vscale x 2 x i64> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: umulh_i64:
; CHECK: umulh z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.umulh.nxv2i64(<vscale x 2 x i1> %pg,
                                                                 <vscale x 2 x i64> %a,
                                                                 <vscale x 2 x i64> %b)
  ret <vscale x 2 x i64> %out
}

; As umulh_i32 but where pg is i8 based and thus compatible for i32.
define <vscale x 4 x i32> @umulh_i32_ptrue_all_b(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: umulh_i32_ptrue_all_b:
; CHECK: umulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.umulh.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                      <vscale x 4 x i32> %a,
                                                                      <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

; As umulh_i32 but where pg is i16 based and thus compatible for i32.
define <vscale x 4 x i32> @umulh_i32_ptrue_all_h(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: umulh_i32_ptrue_all_h:
; CHECK: umulh z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg.h = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %pg.h)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.umulh.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                      <vscale x 4 x i32> %a,
                                                                      <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

; As umulh_i32 but where pg is i64 based, which is not compatibile for i32 and
; thus inactive lanes are important and the immediate form cannot be used.
define <vscale x 4 x i32> @umulh_i32_ptrue_all_d(<vscale x 4 x i32> %a, <vscale x 4 x i32> %b) #0 {
; CHECK-LABEL: umulh_i32_ptrue_all_d:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].d
; CHECK-DAG: umulh z0.s, [[PG]]/m, z0.s, z1.s
; CHECK-NEXT: ret
  %pg.d = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %pg.d)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.umulh.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                      <vscale x 4 x i32> %a,
                                                                      <vscale x 4 x i32> %b)
  ret <vscale x 4 x i32> %out
}

;
; ASR (wide)
;

define <vscale x 16 x i8> @asr_i8(<vscale x 16 x i8> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: asr_i8:
; CHECK: asr z0.b, z0.b, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                    <vscale x 16 x i8> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @asr_i16(<vscale x 8 x i16> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: asr_i16:
; CHECK: asr z0.h, z0.h, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x i16> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @asr_i32(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: asr_i32:
; CHECK: asr z0.s, z0.s, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; LSL (wide)
;

define <vscale x 16 x i8> @lsl_i8(<vscale x 16 x i8> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsl_i8:
; CHECK: lsl z0.b, z0.b, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsl.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                    <vscale x 16 x i8> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsl_i16(<vscale x 8 x i16> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsl_i16:
; CHECK: lsl z0.h, z0.h, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsl.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x i16> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsl_i32(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsl_i32:
; CHECK: lsl z0.s, z0.s, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsl.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; LSR (wide)
;

define <vscale x 16 x i8> @lsr_i8(<vscale x 16 x i8> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsr_i8:
; CHECK: lsr z0.b, z0.b, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.lsr.wide.nxv16i8(<vscale x 16 x i1> %pg,
                                                                    <vscale x 16 x i8> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @lsr_i16(<vscale x 8 x i16> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsr_i16:
; CHECK: lsr z0.h, z0.h, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.lsr.wide.nxv8i16(<vscale x 8 x i1> %pg,
                                                                    <vscale x 8 x i16> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @lsr_i32(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsr_i32:
; CHECK: lsr z0.s, z0.s, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1> %pg,
                                                                    <vscale x 4 x i32> %a,
                                                                    <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

; As lsr_i32 but where pg is i8 based and thus compatible for i32.
define <vscale x 4 x i32> @lsr_i32_ptrue_all_b(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsr_i32_ptrue_all_b:
; CHECK: lsr z0.s, z0.s, z1.d
; CHECK-NEXT: ret
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                         <vscale x 4 x i32> %a,
                                                                         <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

; As lsr_i32 but where pg is i16 based and thus compatible for i32.
define <vscale x 4 x i32> @lsr_i32_ptrue_all_h(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsr_i32_ptrue_all_h:
; CHECK: lsr z0.s, z0.s, z1.d
; CHECK-NEXT: ret
  %pg.h = tail call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1> %pg.h)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                         <vscale x 4 x i32> %a,
                                                                         <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

; As lsr_i32 but where pg is i64 based, which is not compatibile for i32 and
; thus inactive lanes are important and the immediate form cannot be used.
define <vscale x 4 x i32> @lsr_i32_ptrue_all_d(<vscale x 4 x i32> %a, <vscale x 2 x i64> %b) #0 {
; CHECK-LABEL: lsr_i32_ptrue_all_d:
; CHECK-DAG: ptrue [[PG:p[0-9]+]].d
; CHECK-DAG: lsr z0.s, [[PG]]/m, z0.s, z1.d
; CHECK-NEXT: ret
  %pg.d = tail call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %pg.b = tail call <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1> %pg.d)
  %pg.s = tail call <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1> %pg.b)
  %out = tail call <vscale x 4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x 4 x i1> %pg.s,
                                                                         <vscale x 4 x i32> %a,
                                                                         <vscale x 2 x i64> %b)
  ret <vscale x 4 x i32> %out
}

;
; FADD
;

define <vscale x 8 x half> @fadd_half(<vscale x 8 x half> %a, <vscale x 8 x half> %b) #0 {
; CHECK-LABEL: fadd_half:
; CHECK: fadd z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fadd_float(<vscale x 4 x float> %a, <vscale x 4 x float> %b) #0 {
; CHECK-LABEL: fadd_float:
; CHECK: fadd z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fadd_double(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: fadd_double:
; CHECK: fadd z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FSUB
;

define <vscale x 8 x half> @fsub_half(<vscale x 8 x half> %a, <vscale x 8 x half> %b) #0 {
; CHECK-LABEL: fsub_half:
; CHECK: fsub z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fsub.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fsub_float(<vscale x 4 x float> %a, <vscale x 4 x float> %b) #0 {
; CHECK-LABEL: fsub_float:
; CHECK: fsub z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fsub_double(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: fsub_double:
; CHECK: fsub z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

;
; FMUL
;

define <vscale x 8 x half> @fmul_half(<vscale x 8 x half> %a, <vscale x 8 x half> %b) #0 {
; CHECK-LABEL: fmul_half:
; CHECK: fmul z0.h, z0.h, z1.h
; CHECK-NEXT: ret
  %pg = call <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32 31)
  %out = call <vscale x 8 x half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1> %pg,
                                                                 <vscale x 8 x half> %a,
                                                                 <vscale x 8 x half> %b)
  ret <vscale x 8 x half> %out
}

define <vscale x 4 x float> @fmul_float(<vscale x 4 x float> %a, <vscale x 4 x float> %b) #0 {
; CHECK-LABEL: fmul_float:
; CHECK: fmul z0.s, z0.s, z1.s
; CHECK-NEXT: ret
  %pg = call <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32 31)
  %out = call <vscale x 4 x float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1> %pg,
                                                                  <vscale x 4 x float> %a,
                                                                  <vscale x 4 x float> %b)
  ret <vscale x 4 x float> %out
}

define <vscale x 2 x double> @fmul_double(<vscale x 2 x double> %a, <vscale x 2 x double> %b) #0 {
; CHECK-LABEL: fmul_double:
; CHECK: fmul z0.d, z0.d, z1.d
; CHECK-NEXT: ret
  %pg = call <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32 31)
  %out = call <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1> %pg,
                                                                   <vscale x 2 x double> %a,
                                                                   <vscale x 2 x double> %b)
  ret <vscale x 2 x double> %out
}

declare <vscale x 16 x  i8> @llvm.aarch64.sve.mul.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 16 x  i8>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.mul.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x  8 x i16>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.mul.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.mul.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.smulh.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 16 x  i8>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.smulh.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x  8 x i16>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.smulh.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.smulh.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.umulh.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 16 x  i8>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.umulh.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x  8 x i16>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.umulh.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x  4 x i32>)
declare <vscale x  2 x i64> @llvm.aarch64.sve.umulh.nxv2i64(<vscale x  2 x i1>, <vscale x  2 x i64>, <vscale x  2 x i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.asr.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 2 x i64>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.asr.wide.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x 2 x i64>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.asr.wide.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.lsl.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 2 x i64>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.lsl.wide.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x 2 x i64>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.lsl.wide.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x 2 x i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.lsr.wide.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>, <vscale x 2 x i64>)
declare <vscale x  8 x i16> @llvm.aarch64.sve.lsr.wide.nxv8i16(<vscale x  8 x i1>, <vscale x  8 x i16>, <vscale x 2 x i64>)
declare <vscale x  4 x i32> @llvm.aarch64.sve.lsr.wide.nxv4i32(<vscale x  4 x i1>, <vscale x  4 x i32>, <vscale x 2 x i64>)

declare <vscale x 8 x   half> @llvm.aarch64.sve.fadd.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x   half>, <vscale x 8 x   half>)
declare <vscale x 4 x  float> @llvm.aarch64.sve.fadd.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x  float>, <vscale x 4 x  float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fadd.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x   half> @llvm.aarch64.sve.fsub.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x   half>, <vscale x 8 x   half>)
declare <vscale x 4 x  float> @llvm.aarch64.sve.fsub.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x  float>, <vscale x 4 x  float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fsub.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 8 x   half> @llvm.aarch64.sve.fmul.nxv8f16(<vscale x 8 x i1>, <vscale x 8 x   half>, <vscale x 8 x   half>)
declare <vscale x 4 x  float> @llvm.aarch64.sve.fmul.nxv4f32(<vscale x 4 x i1>, <vscale x 4 x  float>, <vscale x 4 x  float>)
declare <vscale x 2 x double> @llvm.aarch64.sve.fmul.nxv2f64(<vscale x 2 x i1>, <vscale x 2 x double>, <vscale x 2 x double>)

declare <vscale x 4 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv4i1(<vscale x 16 x i1>)
declare <vscale x 8 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv8i1(<vscale x 16 x i1>)
declare <vscale x 2 x i1> @llvm.aarch64.sve.convert.from.svbool.nxv2i1(<vscale x 16 x i1>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv4i1(<vscale x 16 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv8i1(<vscale x 8 x i1>)
declare <vscale x 16 x i1> @llvm.aarch64.sve.convert.to.svbool.nxv2i1(<vscale x 2 x i1>)

declare <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32)
declare <vscale x 8 x i1> @llvm.aarch64.sve.ptrue.nxv8i1(i32)
declare <vscale x 4 x i1> @llvm.aarch64.sve.ptrue.nxv4i1(i32)
declare <vscale x 2 x i1> @llvm.aarch64.sve.ptrue.nxv2i1(i32)

attributes #0 = { "target-features"="+sve2" }
