; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; If this check fails please read test/CodeGen/AArch64/README for instructions on how to resolve it.
; WARN-NOT: warning

define <vscale x 16 x i8> @mad_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: mad_i8:
; CHECK: mad z0.b, p0/m, z1.b, z2.b 
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.mad.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b,
                                                               <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @mad_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: mad_i16:
; CHECK: mad z0.h, p0/m, z1.h, z2.h 
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.mad.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b,
                                                               <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @mad_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: mad_i32:
; CHECK: mad z0.s, p0/m, z1.s, z2.s 
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.mad.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b,
                                                               <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @mad_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: mad_i64:
; CHECK: mad z0.d, p0/m, z1.d, z2.d 
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.mad.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b,
                                                               <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

define <vscale x 16 x i8> @msb_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: msb_i8:
; CHECK: msb z0.b, p0/m, z1.b, z2.b 
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.msb.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b,
                                                               <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @msb_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: msb_i16:
; CHECK: msb z0.h, p0/m, z1.h, z2.h 
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.msb.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b,
                                                               <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @msb_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: msb_i32:
; CHECK: msb z0.s, p0/m, z1.s, z2.s 
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.msb.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b,
                                                               <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @msb_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: msb_i64:
; CHECK: msb z0.d, p0/m, z1.d, z2.d 
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.msb.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b,
                                                               <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}


define <vscale x 16 x i8> @mla_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: mla_i8:
; CHECK: mla z0.b, p0/m, z1.b, z2.b 
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.mla.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b,
                                                               <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @mla_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: mla_i16:
; CHECK: mla z0.h, p0/m, z1.h, z2.h 
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.mla.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b,
                                                               <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @mla_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: mla_i32:
; CHECK: mla z0.s, p0/m, z1.s, z2.s 
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.mla.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b,
                                                               <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @mla_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: mla_i64:
; CHECK: mla z0.d, p0/m, z1.d, z2.d 
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.mla.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b,
                                                               <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}


define <vscale x 16 x i8> @mls_i8(<vscale x 16 x i1> %pg, <vscale x 16 x i8> %a, <vscale x 16 x i8> %b, <vscale x 16 x i8> %c) {
; CHECK-LABEL: mls_i8:
; CHECK: mls z0.b, p0/m, z1.b, z2.b 
; CHECK-NEXT: ret
  %out = call <vscale x 16 x i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1> %pg,
                                                               <vscale x 16 x i8> %a,
                                                               <vscale x 16 x i8> %b,
                                                               <vscale x 16 x i8> %c)
  ret <vscale x 16 x i8> %out
}

define <vscale x 8 x i16> @mls_i16(<vscale x 8 x i1> %pg, <vscale x 8 x i16> %a, <vscale x 8 x i16> %b, <vscale x 8 x i16> %c) {
; CHECK-LABEL: mls_i16:
; CHECK: mls z0.h, p0/m, z1.h, z2.h 
; CHECK-NEXT: ret
  %out = call <vscale x 8 x i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1> %pg,
                                                               <vscale x 8 x i16> %a,
                                                               <vscale x 8 x i16> %b,
                                                               <vscale x 8 x i16> %c)
  ret <vscale x 8 x i16> %out
}

define <vscale x 4 x i32> @mls_i32(<vscale x 4 x i1> %pg, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, <vscale x 4 x i32> %c) {
; CHECK-LABEL: mls_i32:
; CHECK: mls z0.s, p0/m, z1.s, z2.s 
; CHECK-NEXT: ret
  %out = call <vscale x 4 x i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1> %pg,
                                                               <vscale x 4 x i32> %a,
                                                               <vscale x 4 x i32> %b,
                                                               <vscale x 4 x i32> %c)
  ret <vscale x 4 x i32> %out
}

define <vscale x 2 x i64> @mls_i64(<vscale x 2 x i1> %pg, <vscale x 2 x i64> %a, <vscale x 2 x i64> %b, <vscale x 2 x i64> %c) {
; CHECK-LABEL: mls_i64:
; CHECK: mls z0.d, p0/m, z1.d, z2.d 
; CHECK-NEXT: ret
  %out = call <vscale x 2 x i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1> %pg,
                                                               <vscale x 2 x i64> %a,
                                                               <vscale x 2 x i64> %b,
                                                               <vscale x 2 x i64> %c)
  ret <vscale x 2 x i64> %out
}

declare <vscale x 16 x  i8> @llvm.aarch64.sve.mad.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>,<vscale x 16 x  i8>,<vscale x 16 x  i8>)
declare <vscale x 8 x  i16> @llvm.aarch64.sve.mad.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x  i16>,<vscale x 8 x  i16>,<vscale x 8 x  i16>)
declare <vscale x 4 x  i32> @llvm.aarch64.sve.mad.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x  i32>,<vscale x 4 x  i32>,<vscale x 4 x  i32>)
declare <vscale x 2 x  i64> @llvm.aarch64.sve.mad.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x  i64>,<vscale x 2 x  i64>,<vscale x 2 x  i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.msb.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>,<vscale x 16 x  i8>,<vscale x 16 x  i8>)
declare <vscale x 8 x  i16> @llvm.aarch64.sve.msb.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x  i16>,<vscale x 8 x  i16>,<vscale x 8 x  i16>)
declare <vscale x 4 x  i32> @llvm.aarch64.sve.msb.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x  i32>,<vscale x 4 x  i32>,<vscale x 4 x  i32>)
declare <vscale x 2 x  i64> @llvm.aarch64.sve.msb.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x  i64>,<vscale x 2 x  i64>,<vscale x 2 x  i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.mla.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>,<vscale x 16 x  i8>,<vscale x 16 x  i8>)
declare <vscale x 8 x  i16> @llvm.aarch64.sve.mla.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x  i16>,<vscale x 8 x  i16>,<vscale x 8 x  i16>)
declare <vscale x 4 x  i32> @llvm.aarch64.sve.mla.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x  i32>,<vscale x 4 x  i32>,<vscale x 4 x  i32>)
declare <vscale x 2 x  i64> @llvm.aarch64.sve.mla.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x  i64>,<vscale x 2 x  i64>,<vscale x 2 x  i64>)

declare <vscale x 16 x  i8> @llvm.aarch64.sve.mls.nxv16i8(<vscale x 16 x i1>, <vscale x 16 x  i8>,<vscale x 16 x  i8>,<vscale x 16 x  i8>)
declare <vscale x 8 x  i16> @llvm.aarch64.sve.mls.nxv8i16(<vscale x 8 x i1>, <vscale x 8 x  i16>,<vscale x 8 x  i16>,<vscale x 8 x  i16>)
declare <vscale x 4 x  i32> @llvm.aarch64.sve.mls.nxv4i32(<vscale x 4 x i1>, <vscale x 4 x  i32>,<vscale x 4 x  i32>,<vscale x 4 x  i32>)
declare <vscale x 2 x  i64> @llvm.aarch64.sve.mls.nxv2i64(<vscale x 2 x i1>, <vscale x 2 x  i64>,<vscale x 2 x  i64>,<vscale x 2 x  i64>)
