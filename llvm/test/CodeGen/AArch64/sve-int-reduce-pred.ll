; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

define i64 @saddv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: saddv_i8:
; CHECK: saddv d[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.saddv.nxv16i8(<vscale x 16 x i1> %pg,
                                                  <vscale x 16 x i8> %a)
  ret i64 %out
}

define i64 @saddv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: saddv_i16:
; CHECK: saddv d[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.saddv.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i64 %out
}


define i64 @saddv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: saddv_i32:
; CHECK: saddv d[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.saddv.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i64 %out
}

define i64 @saddv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: saddv_i64
; CHECK: uaddv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.saddv.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %out
}

define i64 @uaddv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: uaddv_i8:
; CHECK: uaddv d[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uaddv.nxv16i8(<vscale x 16 x i1> %pg,
                                                  <vscale x 16 x i8> %a)
  ret i64 %out
}

define i64 @uaddv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: uaddv_i16:
; CHECK: uaddv d[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uaddv.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i64 %out
}


define i64 @uaddv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: uaddv_i32:
; CHECK: uaddv d[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uaddv.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i64 %out
}

define i64 @uaddv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: uaddv_i64:
; CHECK: uaddv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uaddv.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %out
}

define i8 @smaxv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: smaxv_i8:
; CHECK: smaxv b[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: umov w0, v[[REDUCE]].b[0]
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.smaxv.nxv16i8(<vscale x 16 x i1> %pg,
                                                 <vscale x 16 x i8> %a)
  ret i8 %out
}

define i16 @smaxv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: smaxv_i16:
; CHECK: smaxv h[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: umov w0, v[[REDUCE]].h[0]
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.smaxv.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i16 %out
}

define i32 @smaxv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: smaxv_i32:
; CHECK: smaxv s[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.smaxv.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i32 %out
}

define i64 @smaxv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: smaxv_i64:
; CHECK: smaxv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.smaxv.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %out
}

define i8 @umaxv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: umaxv_i8:
; CHECK: umaxv b[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: umov w0, v[[REDUCE]].b[0]
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.umaxv.nxv16i8(<vscale x 16 x i1> %pg,
                                                 <vscale x 16 x i8> %a)
  ret i8 %out
}

define i16 @umaxv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: umaxv_i16:
; CHECK: umaxv h[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: umov w0, v[[REDUCE]].h[0]
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.umaxv.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i16 %out
}

define i32 @umaxv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: umaxv_i32:
; CHECK: umaxv s[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.umaxv.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i32 %out
}

define i64 @umaxv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: umaxv_i64:
; CHECK: umaxv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.umaxv.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %out
}

define i8 @sminv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: sminv_i8:
; CHECK: sminv b[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: umov w0, v[[REDUCE]].b[0]
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.sminv.nxv16i8(<vscale x 16 x i1> %pg,
                                                 <vscale x 16 x i8> %a)
  ret i8 %out
}

define i16 @sminv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: sminv_i16:
; CHECK: sminv h[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: umov w0, v[[REDUCE]].h[0]
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.sminv.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i16 %out
}

define i32 @sminv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: sminv_i32:
; CHECK: sminv s[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.sminv.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i32 %out
}

define i64 @sminv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: sminv_i64:
; CHECK: sminv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.sminv.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %out
}

define i8 @uminv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: uminv_i8:
; CHECK: uminv b[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: umov w0, v[[REDUCE]].b[0]
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.uminv.nxv16i8(<vscale x 16 x i1> %pg,
                                                 <vscale x 16 x i8> %a)
  ret i8 %out
}

define i16 @uminv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: uminv_i16:
; CHECK: uminv h[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: umov w0, v[[REDUCE]].h[0]
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.uminv.nxv8i16(<vscale x 8 x i1> %pg,
                                                  <vscale x 8 x i16> %a)
  ret i16 %out
}

define i32 @uminv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: uminv_i32:
; CHECK: uminv s[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.uminv.nxv4i32(<vscale x 4 x i1> %pg,
                                                  <vscale x 4 x i32> %a)
  ret i32 %out
}

define i64 @uminv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: uminv_i64:
; CHECK: uminv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.uminv.nxv2i64(<vscale x 2 x i1> %pg,
                                                  <vscale x 2 x i64> %a)
  ret i64 %out
}

define i8 @orv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: orv_i8:
; CHECK: orv b[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: umov w0, v[[REDUCE]].b[0]
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.orv.nxv16i8(<vscale x 16 x i1> %pg,
                                               <vscale x 16 x i8> %a)
  ret i8 %out
}

define i16 @orv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: orv_i16:
; CHECK: orv h[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: umov w0, v[[REDUCE]].h[0]
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.orv.nxv8i16(<vscale x 8 x i1> %pg,
                                                <vscale x 8 x i16> %a)
  ret i16 %out
}

define i32 @orv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: orv_i32:
; CHECK: orv s[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.orv.nxv4i32(<vscale x 4 x i1> %pg,
                                                <vscale x 4 x i32> %a)
  ret i32 %out
}

define i64 @orv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: orv_i64:
; CHECK: orv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.orv.nxv2i64(<vscale x 2 x i1> %pg,
                                                <vscale x 2 x i64> %a)
  ret i64 %out
}

define i8 @eorv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: eorv_i8:
; CHECK: eorv b[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: umov w0, v[[REDUCE]].b[0]
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.eorv.nxv16i8(<vscale x 16 x i1> %pg,
                                                <vscale x 16 x i8> %a)
  ret i8 %out
}

define i16 @eorv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: eorv_i16:
; CHECK: eorv h[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: umov w0, v[[REDUCE]].h[0]
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.eorv.nxv8i16(<vscale x 8 x i1> %pg,
                                                 <vscale x 8 x i16> %a)
  ret i16 %out
}

define i32 @eorv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: eorv_i32:
; CHECK: eorv s[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.eorv.nxv4i32(<vscale x 4 x i1> %pg,
                                                 <vscale x 4 x i32> %a)
  ret i32 %out
}

define i64 @eorv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: eorv_i64:
; CHECK: eorv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.eorv.nxv2i64(<vscale x 2 x i1> %pg,
                                                 <vscale x 2 x i64> %a)
  ret i64 %out
}

define i8 @andv_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a) {
; CHECK-LABEL: andv_i8:
; CHECK: andv b[[REDUCE:[0-9]+]], p0, z0.b
; CHECK: umov w0, v[[REDUCE]].b[0]
; CHECK-NEXT: ret
  %out = call i8 @llvm.aarch64.sve.andv.nxv16i8(<vscale x 16 x i1> %pg,
                                                <vscale x 16 x i8> %a)
  ret i8 %out
}

define i16 @andv_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a) {
; CHECK-LABEL: andv_i16:
; CHECK: andv h[[REDUCE:[0-9]+]], p0, z0.h
; CHECK: umov w0, v[[REDUCE]].h[0]
; CHECK-NEXT: ret
  %out = call i16 @llvm.aarch64.sve.andv.nxv8i16(<vscale x 8 x i1> %pg,
                                                 <vscale x 8 x i16> %a)
  ret i16 %out
}

define i32 @andv_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a) {
; CHECK-LABEL: andv_i32:
; CHECK: andv s[[REDUCE:[0-9]+]], p0, z0.s
; CHECK: fmov w0, s[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i32 @llvm.aarch64.sve.andv.nxv4i32(<vscale x 4 x i1> %pg,
                                                 <vscale x 4 x i32> %a)
  ret i32 %out
}

define i64 @andv_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a) {
; CHECK-LABEL: andv_i64:
; CHECK: andv d[[REDUCE:[0-9]+]], p0, z0.d
; CHECK: fmov x0, d[[REDUCE]]
; CHECK-NEXT: ret
  %out = call i64 @llvm.aarch64.sve.andv.nxv2i64(<vscale x 2 x i1> %pg,
                                                 <vscale x 2 x i64> %a)
  ret i64 %out
}

declare i64 @llvm.aarch64.sve.saddv.nxv16i8(<vscale x 16 x i1>, <vscale x  16 x i8>)
declare i64 @llvm.aarch64.sve.saddv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i64 @llvm.aarch64.sve.saddv.nxv4i32(<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.saddv.nxv2i64(<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i64 @llvm.aarch64.sve.uaddv.nxv16i8(<vscale x 16 x i1>, <vscale x  16 x i8>)
declare i64 @llvm.aarch64.sve.uaddv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i64 @llvm.aarch64.sve.uaddv.nxv4i32(<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.uaddv.nxv2i64(<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i8 @llvm.aarch64.sve.smaxv.nxv16i8(<vscale x 16 x  i1>, <vscale x  16 x  i8>)
declare i16 @llvm.aarch64.sve.smaxv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i32 @llvm.aarch64.sve.smaxv.nxv4i32(<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.smaxv.nxv2i64(<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i8 @llvm.aarch64.sve.umaxv.nxv16i8(<vscale x 16 x  i1>, <vscale x  16 x  i8>)
declare i16 @llvm.aarch64.sve.umaxv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i32 @llvm.aarch64.sve.umaxv.nxv4i32(<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.umaxv.nxv2i64(<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i8 @llvm.aarch64.sve.sminv.nxv16i8(<vscale x 16 x  i1>, <vscale x  16 x  i8>)
declare i16 @llvm.aarch64.sve.sminv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i32 @llvm.aarch64.sve.sminv.nxv4i32(<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.sminv.nxv2i64(<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i8 @llvm.aarch64.sve.uminv.nxv16i8(<vscale x 16 x  i1>, <vscale x  16 x  i8>)
declare i16 @llvm.aarch64.sve.uminv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i32 @llvm.aarch64.sve.uminv.nxv4i32(<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.uminv.nxv2i64(<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i8 @llvm.aarch64.sve.orv.nxv16i8(<vscale x 16 x  i1>, <vscale x  16 x  i8>)
declare i16 @llvm.aarch64.sve.orv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i32 @llvm.aarch64.sve.orv.nxv4i32  (<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.orv.nxv2i64  (<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i8 @llvm.aarch64.sve.eorv.nxv16i8(<vscale x 16 x  i1>, <vscale x  16 x  i8>)
declare i16 @llvm.aarch64.sve.eorv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i32 @llvm.aarch64.sve.eorv.nxv4i32 (<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.eorv.nxv2i64 (<vscale x 2 x  i1>, <vscale x  2 x  i64>)
declare i8 @llvm.aarch64.sve.andv.nxv16i8(<vscale x 16 x  i1>, <vscale x  16 x  i8>)
declare i16 @llvm.aarch64.sve.andv.nxv8i16(<vscale x 8 x  i1>, <vscale x  8 x  i16>)
declare i32 @llvm.aarch64.sve.andv.nxv4i32 (<vscale x 4 x  i1>, <vscale x  4 x  i32>)
declare i64 @llvm.aarch64.sve.andv.nxv2i64 (<vscale x 2 x  i1>, <vscale x  2 x  i64>)
