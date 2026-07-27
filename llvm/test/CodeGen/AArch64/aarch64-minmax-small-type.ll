; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs < %s | FileCheck %s

; Check that small fixed-length integer min/max operations are widened in
; registers to legal NEON types while retaining the original narrow memory
; accesses.

define void @smin_v2i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: smin_v2i8:
; CHECK:       ldr h[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr h[[B:[0-9]+]], [x2]
; CHECK-NEXT:  smin v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str h[[A]], [x0]
  %va = load <2 x i8>, ptr %a, align 1
  %vb = load <2 x i8>, ptr %b, align 1
  %min = call <2 x i8> @llvm.smin.v2i8(<2 x i8> %va, <2 x i8> %vb)
  store <2 x i8> %min, ptr %dst, align 1
  ret void
}

define void @smax_v2i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: smax_v2i8:
; CHECK:       ldr h[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr h[[B:[0-9]+]], [x2]
; CHECK-NEXT:  smax v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str h[[A]], [x0]
  %va = load <2 x i8>, ptr %a, align 1
  %vb = load <2 x i8>, ptr %b, align 1
  %max = call <2 x i8> @llvm.smax.v2i8(<2 x i8> %va, <2 x i8> %vb)
  store <2 x i8> %max, ptr %dst, align 1
  ret void
}

define void @umin_v2i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: umin_v2i8:
; CHECK:       ldr h[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr h[[B:[0-9]+]], [x2]
; CHECK-NEXT:  umin v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str h[[A]], [x0]
  %va = load <2 x i8>, ptr %a, align 1
  %vb = load <2 x i8>, ptr %b, align 1
  %min = call <2 x i8> @llvm.umin.v2i8(<2 x i8> %va, <2 x i8> %vb)
  store <2 x i8> %min, ptr %dst, align 1
  ret void
}

define void @umax_v2i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: umax_v2i8:
; CHECK:       ldr h[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr h[[B:[0-9]+]], [x2]
; CHECK-NEXT:  umax v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str h[[A]], [x0]
  %va = load <2 x i8>, ptr %a, align 1
  %vb = load <2 x i8>, ptr %b, align 1
  %max = call <2 x i8> @llvm.umax.v2i8(<2 x i8> %va, <2 x i8> %vb)
  store <2 x i8> %max, ptr %dst, align 1
  ret void
}

define void @smin_v4i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: smin_v4i8:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  smin v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <4 x i8>, ptr %a, align 1
  %vb = load <4 x i8>, ptr %b, align 1
  %min = call <4 x i8> @llvm.smin.v4i8(<4 x i8> %va, <4 x i8> %vb)
  store <4 x i8> %min, ptr %dst, align 1
  ret void
}

define void @smax_v4i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: smax_v4i8:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  smax v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <4 x i8>, ptr %a, align 1
  %vb = load <4 x i8>, ptr %b, align 1
  %max = call <4 x i8> @llvm.smax.v4i8(<4 x i8> %va, <4 x i8> %vb)
  store <4 x i8> %max, ptr %dst, align 1
  ret void
}

define void @umin_v4i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: umin_v4i8:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  umin v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <4 x i8>, ptr %a, align 1
  %vb = load <4 x i8>, ptr %b, align 1
  %min = call <4 x i8> @llvm.umin.v4i8(<4 x i8> %va, <4 x i8> %vb)
  store <4 x i8> %min, ptr %dst, align 1
  ret void
}

define void @umax_v4i8(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: umax_v4i8:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  umax v[[A]].8b, v[[A]].8b, v[[B]].8b
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <4 x i8>, ptr %a, align 1
  %vb = load <4 x i8>, ptr %b, align 1
  %max = call <4 x i8> @llvm.umax.v4i8(<4 x i8> %va, <4 x i8> %vb)
  store <4 x i8> %max, ptr %dst, align 1
  ret void
}

define void @smin_v2i16(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: smin_v2i16:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  smin v[[A]].4h, v[[A]].4h, v[[B]].4h
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <2 x i16>, ptr %a, align 2
  %vb = load <2 x i16>, ptr %b, align 2
  %min = call <2 x i16> @llvm.smin.v2i16(<2 x i16> %va, <2 x i16> %vb)
  store <2 x i16> %min, ptr %dst, align 2
  ret void
}

define void @smax_v2i16(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: smax_v2i16:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  smax v[[A]].4h, v[[A]].4h, v[[B]].4h
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <2 x i16>, ptr %a, align 2
  %vb = load <2 x i16>, ptr %b, align 2
  %max = call <2 x i16> @llvm.smax.v2i16(<2 x i16> %va, <2 x i16> %vb)
  store <2 x i16> %max, ptr %dst, align 2
  ret void
}

define void @umin_v2i16(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: umin_v2i16:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  umin v[[A]].4h, v[[A]].4h, v[[B]].4h
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <2 x i16>, ptr %a, align 2
  %vb = load <2 x i16>, ptr %b, align 2
  %min = call <2 x i16> @llvm.umin.v2i16(<2 x i16> %va, <2 x i16> %vb)
  store <2 x i16> %min, ptr %dst, align 2
  ret void
}

define void @umax_v2i16(ptr %dst, ptr %a, ptr %b) {
; CHECK-LABEL: umax_v2i16:
; CHECK:       ldr s[[A:[0-9]+]], [x1]
; CHECK-NEXT:  ldr s[[B:[0-9]+]], [x2]
; CHECK-NEXT:  umax v[[A]].4h, v[[A]].4h, v[[B]].4h
; CHECK-NEXT:  str s[[A]], [x0]
  %va = load <2 x i16>, ptr %a, align 2
  %vb = load <2 x i16>, ptr %b, align 2
  %max = call <2 x i16> @llvm.umax.v2i16(<2 x i16> %va, <2 x i16> %vb)
  store <2 x i16> %max, ptr %dst, align 2
  ret void
}

declare <2 x i8> @llvm.smin.v2i8(<2 x i8>, <2 x i8>)
declare <2 x i8> @llvm.smax.v2i8(<2 x i8>, <2 x i8>)
declare <2 x i8> @llvm.umin.v2i8(<2 x i8>, <2 x i8>)
declare <2 x i8> @llvm.umax.v2i8(<2 x i8>, <2 x i8>)

declare <4 x i8> @llvm.smin.v4i8(<4 x i8>, <4 x i8>)
declare <4 x i8> @llvm.smax.v4i8(<4 x i8>, <4 x i8>)
declare <4 x i8> @llvm.umin.v4i8(<4 x i8>, <4 x i8>)
declare <4 x i8> @llvm.umax.v4i8(<4 x i8>, <4 x i8>)

declare <2 x i16> @llvm.smin.v2i16(<2 x i16>, <2 x i16>)
declare <2 x i16> @llvm.smax.v2i16(<2 x i16>, <2 x i16>)
declare <2 x i16> @llvm.umin.v2i16(<2 x i16>, <2 x i16>)
declare <2 x i16> @llvm.umax.v2i16(<2 x i16>, <2 x i16>)
