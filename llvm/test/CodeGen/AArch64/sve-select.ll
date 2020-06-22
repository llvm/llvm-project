; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>%t | FileCheck %s
; RUN: FileCheck --check-prefix=WARN --allow-empty %s <%t

; WARN-NOT: warning

; Integer vector select

define <vscale x 16 x i8> @sel_nxv16i8(<vscale x 16 x i1> %p,
                                       <vscale x 16 x i8> %dst,
                                       <vscale x 16 x i8> %a) {
; CHECK-LABEL: sel_nxv16i8:
; CHECK:         mov z0.b, p0/m, z1.b
; CHECK-NEXT:    ret
  %sel = select <vscale x 16 x i1> %p, <vscale x 16 x i8> %a, <vscale x 16 x i8> %dst
  ret <vscale x 16 x i8> %sel
}

define <vscale x 8 x i16> @sel_nxv8i16(<vscale x 8 x i1> %p,
                                       <vscale x 8 x i16> %dst,
                                       <vscale x 8 x i16> %a) {
; CHECK-LABEL: sel_nxv8i16:
; CHECK:         mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x i16> %a, <vscale x 8 x i16> %dst
  ret <vscale x 8 x i16> %sel
}

define <vscale x 4 x i32> @sel_nxv4i32(<vscale x 4 x i1> %p,
                                       <vscale x 4 x i32> %dst,
                                       <vscale x 4 x i32> %a) {
; CHECK-LABEL: sel_nxv4i32:
; CHECK:         mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x i32> %a, <vscale x 4 x i32> %dst
  ret <vscale x 4 x i32> %sel
}

define <vscale x 2 x i64> @sel_nxv2i64(<vscale x 2 x i1> %p,
                                       <vscale x 2 x i64> %dst,
                                       <vscale x 2 x i64> %a) {
; CHECK-LABEL: sel_nxv2i64:
; CHECK:         mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x i64> %a, <vscale x 2 x i64> %dst
  ret <vscale x 2 x i64> %sel
}

; Floating point vector select

define <vscale x 8 x half> @sel_nxv8f16(<vscale x 8 x i1> %p,
                                        <vscale x 8 x half> %dst,
                                        <vscale x 8 x half> %a) {
; CHECK-LABEL: sel_nxv8f16:
; CHECK:         mov z0.h, p0/m, z1.h
; CHECK-NEXT:    ret
  %sel = select <vscale x 8 x i1> %p, <vscale x 8 x half> %a, <vscale x 8 x half> %dst
  ret <vscale x 8 x half> %sel
}

define <vscale x 4 x float> @sel_nxv4f32(<vscale x 4 x i1> %p,
                                         <vscale x 4 x float> %dst,
                                         <vscale x 4 x float> %a) {
; CHECK-LABEL: sel_nxv4f32:
; CHECK:         mov z0.s, p0/m, z1.s
; CHECK-NEXT:    ret
  %sel = select <vscale x 4 x i1> %p, <vscale x 4 x float> %a, <vscale x 4 x float> %dst
  ret <vscale x 4 x float> %sel
}

define <vscale x 2 x float> @sel_nxv2f32(<vscale x 2 x i1> %p,
                                         <vscale x 2 x float> %dst,
                                         <vscale x 2 x float> %a) {
; CHECK-LABEL: sel_nxv2f32:
; CHECK:         mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x float> %a, <vscale x 2 x float> %dst
  ret <vscale x 2 x float> %sel
}

define <vscale x 2 x double> @sel_nxv8f64(<vscale x 2 x i1> %p,
                                          <vscale x 2 x double> %dst,
                                          <vscale x 2 x double> %a) {
; CHECK-LABEL: sel_nxv8f64:
; CHECK:         mov z0.d, p0/m, z1.d
; CHECK-NEXT:    ret
  %sel = select <vscale x 2 x i1> %p, <vscale x 2 x double> %a, <vscale x 2 x double> %dst
  ret <vscale x 2 x double> %sel
}
