; RUN: opt < %s -mtriple=aarch64--linux-gnu -cost-model -analyze | FileCheck %s --check-prefix=COST
; RUN: llc < %s -mtriple=aarch64--linux-gnu | FileCheck %s --check-prefix=CODE

; COST-LABEL: umin.v8i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i8> @llvm.umin.v8i8(<8 x i8> %v0, <8 x i8> %v1)

; CODE-LABEL: umin.v8i8
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.8b, v{{.*}}.8b, v{{.*}}.8b
; CODE-NEXT:   ret

declare <8 x i8> @llvm.umin.v8i8(<8 x i8>, <8 x i8>)
define <8 x i8> @umin.v8i8(<8 x i8> %v0, <8 x i8> %v1) {
  %res = call <8 x i8> @llvm.umin.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  ret <8 x i8> %res
}

; COST-LABEL: umin.v9i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <9 x i8> @llvm.umin.v9i8(<9 x i8> %v0, <9 x i8> %v1)

; CODE-LABEL: umin.v9i8
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <9 x i8> @llvm.umin.v9i8(<9 x i8>, <9 x i8>)
define <9 x i8> @umin.v9i8(<9 x i8> %v0, <9 x i8> %v1) {
  %res = call <9 x i8> @llvm.umin.v9i8(<9 x i8> %v0, <9 x i8> %v1)
  ret <9 x i8> %res
}

; COST-LABEL: umin.v4i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i16> @llvm.umin.v4i16(<4 x i16> %v0, <4 x i16> %v1)

; CODE-LABEL: umin.v4i16
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.4h, v{{.*}}.4h, v{{.*}}.4h
; CODE-NEXT:   ret

declare <4 x i16> @llvm.umin.v4i16(<4 x i16>, <4 x i16>)
define <4 x i16> @umin.v4i16(<4 x i16> %v0, <4 x i16> %v1) {
  %res = call <4 x i16> @llvm.umin.v4i16(<4 x i16> %v0, <4 x i16> %v1)
  ret <4 x i16> %res
}

; COST-LABEL: umin.v16i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <16 x i8> @llvm.umin.v16i8(<16 x i8> %v0, <16 x i8> %v1)

; CODE-LABEL: umin.v16i8
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <16 x i8> @llvm.umin.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @umin.v16i8(<16 x i8> %v0, <16 x i8> %v1) {
  %res = call <16 x i8> @llvm.umin.v16i8(<16 x i8> %v0, <16 x i8> %v1)
  ret <16 x i8> %res
}

; COST-LABEL: umin.v8i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i16> @llvm.umin.v8i16(<8 x i16> %v0, <8 x i16> %v1)

; CODE-LABEL: umin.v8i16
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.8h, v{{.*}}.8h, v{{.*}}.8h
; CODE-NEXT:   ret

declare <8 x i16> @llvm.umin.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @umin.v8i16(<8 x i16> %v0, <8 x i16> %v1) {
  %res = call <8 x i16> @llvm.umin.v8i16(<8 x i16> %v0, <8 x i16> %v1)
  ret <8 x i16> %res
}

; COST-LABEL: umin.v2i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <2 x i32> @llvm.umin.v2i32(<2 x i32> %v0, <2 x i32> %v1)

; CODE-LABEL: umin.v2i32
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.2s, v{{.*}}.2s, v{{.*}}.2s
; CODE-NEXT:   ret

declare <2 x i32> @llvm.umin.v2i32(<2 x i32>, <2 x i32>)
define <2 x i32> @umin.v2i32(<2 x i32> %v0, <2 x i32> %v1) {
  %res = call <2 x i32> @llvm.umin.v2i32(<2 x i32> %v0, <2 x i32> %v1)
  ret <2 x i32> %res
}

; COST-LABEL: umin.v4i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i32> @llvm.umin.v4i32(<4 x i32> %v0, <4 x i32> %v1)

; CODE-LABEL: umin.v4i32
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <4 x i32> @llvm.umin.v4i32(<4 x i32>, <4 x i32>)
define <4 x i32> @umin.v4i32(<4 x i32> %v0, <4 x i32> %v1) {
  %res = call <4 x i32> @llvm.umin.v4i32(<4 x i32> %v0, <4 x i32> %v1)
  ret <4 x i32> %res
}

; COST-LABEL: umin.v8i32
; COST-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %res = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %v0, <8 x i32> %v1)

; CODE-LABEL: umin.v8i32
; CODE:       bb.0
; CODE-NEXT:   umin v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   umin v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <8 x i32> @llvm.umin.v8i32(<8 x i32>, <8 x i32>)
define <8 x i32> @umin.v8i32(<8 x i32> %v0, <8 x i32> %v1) {
  %res = call <8 x i32> @llvm.umin.v8i32(<8 x i32> %v0, <8 x i32> %v1)
  ret <8 x i32> %res
}

; COST-LABEL: umin.v2i64
; COST-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res = call <2 x i64> @llvm.umin.v2i64(<2 x i64> %v0, <2 x i64> %v1)

; CODE-LABEL: umin.v2i64
; CODE:       bb.0
; CODE:        csel
; CODE:        csel

declare <2 x i64> @llvm.umin.v2i64(<2 x i64>, <2 x i64>)
define <2 x i64> @umin.v2i64(<2 x i64> %v0, <2 x i64> %v1) {
  %res = call <2 x i64> @llvm.umin.v2i64(<2 x i64> %v0, <2 x i64> %v1)
  ret <2 x i64> %res
}

; COST-LABEL: smin.v8i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i8> @llvm.smin.v8i8(<8 x i8> %v0, <8 x i8> %v1)

; CODE-LABEL: smin.v8i8
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.8b, v{{.*}}.8b, v{{.*}}.8b
; CODE-NEXT:   ret

declare <8 x i8> @llvm.smin.v8i8(<8 x i8>, <8 x i8>)
define <8 x i8> @smin.v8i8(<8 x i8> %v0, <8 x i8> %v1) {
  %res = call <8 x i8> @llvm.smin.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  ret <8 x i8> %res
}

; COST-LABEL: smin.v9i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <9 x i8> @llvm.smin.v9i8(<9 x i8> %v0, <9 x i8> %v1)

; CODE-LABEL: smin.v9i8
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <9 x i8> @llvm.smin.v9i8(<9 x i8>, <9 x i8>)
define <9 x i8> @smin.v9i8(<9 x i8> %v0, <9 x i8> %v1) {
  %res = call <9 x i8> @llvm.smin.v9i8(<9 x i8> %v0, <9 x i8> %v1)
  ret <9 x i8> %res
}

; COST-LABEL: smin.v16i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %v0, <16 x i8> %v1)

; CODE-LABEL: smin.v16i8
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <16 x i8> @llvm.smin.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @smin.v16i8(<16 x i8> %v0, <16 x i8> %v1) {
  %res = call <16 x i8> @llvm.smin.v16i8(<16 x i8> %v0, <16 x i8> %v1)
  ret <16 x i8> %res
}

; COST-LABEL: smin.v4i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i16> @llvm.smin.v4i16(<4 x i16> %v0, <4 x i16> %v1)

; CODE-LABEL: smin.v4i16
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.4h, v{{.*}}.4h, v{{.*}}.4h
; CODE-NEXT:   ret

declare <4 x i16> @llvm.smin.v4i16(<4 x i16>, <4 x i16>)
define <4 x i16> @smin.v4i16(<4 x i16> %v0, <4 x i16> %v1) {
  %res = call <4 x i16> @llvm.smin.v4i16(<4 x i16> %v0, <4 x i16> %v1)
  ret <4 x i16> %res
}

; COST-LABEL: smin.v8i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i16> @llvm.smin.v8i16(<8 x i16> %v0, <8 x i16> %v1)

; CODE-LABEL: smin.v8i16
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.8h, v{{.*}}.8h, v{{.*}}.8h
; CODE-NEXT:   ret

declare <8 x i16> @llvm.smin.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @smin.v8i16(<8 x i16> %v0, <8 x i16> %v1) {
  %res = call <8 x i16> @llvm.smin.v8i16(<8 x i16> %v0, <8 x i16> %v1)
  ret <8 x i16> %res
}

; COST-LABEL: smin.v2i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <2 x i32> @llvm.smin.v2i32(<2 x i32> %v0, <2 x i32> %v1)

; CODE-LABEL: smin.v2i32
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.2s, v{{.*}}.2s, v{{.*}}.2s
; CODE-NEXT:   ret

declare <2 x i32> @llvm.smin.v2i32(<2 x i32>, <2 x i32>)
define <2 x i32> @smin.v2i32(<2 x i32> %v0, <2 x i32> %v1) {
  %res = call <2 x i32> @llvm.smin.v2i32(<2 x i32> %v0, <2 x i32> %v1)
  ret <2 x i32> %res
}

; COST-LABEL: smin.v4i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %v0, <4 x i32> %v1)

; CODE-LABEL: smin.v4i32
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <4 x i32> @llvm.smin.v4i32(<4 x i32>, <4 x i32>)
define <4 x i32> @smin.v4i32(<4 x i32> %v0, <4 x i32> %v1) {
  %res = call <4 x i32> @llvm.smin.v4i32(<4 x i32> %v0, <4 x i32> %v1)
  ret <4 x i32> %res
}

; COST-LABEL: smin.v8i32
; COST-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %res = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %v0, <8 x i32> %v1)

; CODE-LABEL: smin.v8i32
; CODE:       bb.0
; CODE-NEXT:   smin v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   smin v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <8 x i32> @llvm.smin.v8i32(<8 x i32>, <8 x i32>)
define <8 x i32> @smin.v8i32(<8 x i32> %v0, <8 x i32> %v1) {
  %res = call <8 x i32> @llvm.smin.v8i32(<8 x i32> %v0, <8 x i32> %v1)
  ret <8 x i32> %res
}

; COST-LABEL: smin.v2i64
; COST-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res = call <2 x i64> @llvm.smin.v2i64(<2 x i64> %v0, <2 x i64> %v1)

; CODE-LABEL: smin.v2i64
; CODE:       bb.0
; CODE:        csel
; CODE:        csel

declare <2 x i64> @llvm.smin.v2i64(<2 x i64>, <2 x i64>)
define <2 x i64> @smin.v2i64(<2 x i64> %v0, <2 x i64> %v1) {
  %res = call <2 x i64> @llvm.smin.v2i64(<2 x i64> %v0, <2 x i64> %v1)
  ret <2 x i64> %res
}

; COST-LABEL: umax.v8i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i8> @llvm.umax.v8i8(<8 x i8> %v0, <8 x i8> %v1)

; CODE-LABEL: umax.v8i8
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.8b, v{{.*}}.8b, v{{.*}}.8b
; CODE-NEXT:   ret

declare <8 x i8> @llvm.umax.v8i8(<8 x i8>, <8 x i8>)
define <8 x i8> @umax.v8i8(<8 x i8> %v0, <8 x i8> %v1) {
  %res = call <8 x i8> @llvm.umax.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  ret <8 x i8> %res
}

; COST-LABEL: umax.v9i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <9 x i8> @llvm.umax.v9i8(<9 x i8> %v0, <9 x i8> %v1)

; CODE-LABEL: umax.v9i8
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <9 x i8> @llvm.umax.v9i8(<9 x i8>, <9 x i8>)
define <9 x i8> @umax.v9i8(<9 x i8> %v0, <9 x i8> %v1) {
  %res = call <9 x i8> @llvm.umax.v9i8(<9 x i8> %v0, <9 x i8> %v1)
  ret <9 x i8> %res
}

; COST-LABEL: umax.v16i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <16 x i8> @llvm.umax.v16i8(<16 x i8> %v0, <16 x i8> %v1)

; CODE-LABEL: umax.v16i8
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <16 x i8> @llvm.umax.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @umax.v16i8(<16 x i8> %v0, <16 x i8> %v1) {
  %res = call <16 x i8> @llvm.umax.v16i8(<16 x i8> %v0, <16 x i8> %v1)
  ret <16 x i8> %res
}

; COST-LABEL: umax.v4i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i16> @llvm.umax.v4i16(<4 x i16> %v0, <4 x i16> %v1)

; CODE-LABEL: umax.v4i16
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.4h, v{{.*}}.4h, v{{.*}}.4h
; CODE-NEXT:   ret

declare <4 x i16> @llvm.umax.v4i16(<4 x i16>, <4 x i16>)
define <4 x i16> @umax.v4i16(<4 x i16> %v0, <4 x i16> %v1) {
  %res = call <4 x i16> @llvm.umax.v4i16(<4 x i16> %v0, <4 x i16> %v1)
  ret <4 x i16> %res
}

; COST-LABEL: umax.v8i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i16> @llvm.umax.v8i16(<8 x i16> %v0, <8 x i16> %v1)

; CODE-LABEL: umax.v8i16
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.8h, v{{.*}}.8h, v{{.*}}.8h
; CODE-NEXT:   ret

declare <8 x i16> @llvm.umax.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @umax.v8i16(<8 x i16> %v0, <8 x i16> %v1) {
  %res = call <8 x i16> @llvm.umax.v8i16(<8 x i16> %v0, <8 x i16> %v1)
  ret <8 x i16> %res
}

; COST-LABEL: umax.v2i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <2 x i32> @llvm.umax.v2i32(<2 x i32> %v0, <2 x i32> %v1)

; CODE-LABEL: umax.v2i32
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.2s, v{{.*}}.2s, v{{.*}}.2s
; CODE-NEXT:   ret

declare <2 x i32> @llvm.umax.v2i32(<2 x i32>, <2 x i32>)
define <2 x i32> @umax.v2i32(<2 x i32> %v0, <2 x i32> %v1) {
  %res = call <2 x i32> @llvm.umax.v2i32(<2 x i32> %v0, <2 x i32> %v1)
  ret <2 x i32> %res
}

; COST-LABEL: umax.v4i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i32> @llvm.umax.v4i32(<4 x i32> %v0, <4 x i32> %v1)

; CODE-LABEL: umax.v4i32
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <4 x i32> @llvm.umax.v4i32(<4 x i32>, <4 x i32>)
define <4 x i32> @umax.v4i32(<4 x i32> %v0, <4 x i32> %v1) {
  %res = call <4 x i32> @llvm.umax.v4i32(<4 x i32> %v0, <4 x i32> %v1)
  ret <4 x i32> %res
}

; COST-LABEL: umax.v8i32
; COST-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %res = call <8 x i32> @llvm.umax.v8i32(<8 x i32> %v0, <8 x i32> %v1)

; CODE-LABEL: umax.v8i32
; CODE:       bb.0
; CODE-NEXT:   umax v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   umax v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <8 x i32> @llvm.umax.v8i32(<8 x i32>, <8 x i32>)
define <8 x i32> @umax.v8i32(<8 x i32> %v0, <8 x i32> %v1) {
  %res = call <8 x i32> @llvm.umax.v8i32(<8 x i32> %v0, <8 x i32> %v1)
  ret <8 x i32> %res
}

; COST-LABEL: umax.v2i64
; COST-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res = call <2 x i64> @llvm.umax.v2i64(<2 x i64> %v0, <2 x i64> %v1)

; CODE-LABEL: umax.v2i64
; CODE:       bb.0
; CODE:        csel
; CODE:        csel

declare <2 x i64> @llvm.umax.v2i64(<2 x i64>, <2 x i64>)
define <2 x i64> @umax.v2i64(<2 x i64> %v0, <2 x i64> %v1) {
  %res = call <2 x i64> @llvm.umax.v2i64(<2 x i64> %v0, <2 x i64> %v1)
  ret <2 x i64> %res
}

; COST-LABEL: smax.v8i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i8> @llvm.smax.v8i8(<8 x i8> %v0, <8 x i8> %v1)

; CODE-LABEL: smax.v8i8
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.8b, v{{.*}}.8b, v{{.*}}.8b
; CODE-NEXT:   ret

declare <8 x i8> @llvm.smax.v8i8(<8 x i8>, <8 x i8>)
define <8 x i8> @smax.v8i8(<8 x i8> %v0, <8 x i8> %v1) {
  %res = call <8 x i8> @llvm.smax.v8i8(<8 x i8> %v0, <8 x i8> %v1)
  ret <8 x i8> %res
}

; COST-LABEL: smax.v9i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <9 x i8> @llvm.smax.v9i8(<9 x i8> %v0, <9 x i8> %v1)

; CODE-LABEL: smax.v9i8
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <9 x i8> @llvm.smax.v9i8(<9 x i8>, <9 x i8>)
define <9 x i8> @smax.v9i8(<9 x i8> %v0, <9 x i8> %v1) {
  %res = call <9 x i8> @llvm.smax.v9i8(<9 x i8> %v0, <9 x i8> %v1)
  ret <9 x i8> %res
}

; COST-LABEL: smax.v16i8
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <16 x i8> @llvm.smax.v16i8(<16 x i8> %v0, <16 x i8> %v1)

; CODE-LABEL: smax.v16i8
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.16b, v{{.*}}.16b, v{{.*}}.16b
; CODE-NEXT:   ret

declare <16 x i8> @llvm.smax.v16i8(<16 x i8>, <16 x i8>)
define <16 x i8> @smax.v16i8(<16 x i8> %v0, <16 x i8> %v1) {
  %res = call <16 x i8> @llvm.smax.v16i8(<16 x i8> %v0, <16 x i8> %v1)
  ret <16 x i8> %res
}

; COST-LABEL: smax.v4i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i16> @llvm.smax.v4i16(<4 x i16> %v0, <4 x i16> %v1)

; CODE-LABEL: smax.v4i16
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.4h, v{{.*}}.4h, v{{.*}}.4h
; CODE-NEXT:   ret

declare <4 x i16> @llvm.smax.v4i16(<4 x i16>, <4 x i16>)
define <4 x i16> @smax.v4i16(<4 x i16> %v0, <4 x i16> %v1) {
  %res = call <4 x i16> @llvm.smax.v4i16(<4 x i16> %v0, <4 x i16> %v1)
  ret <4 x i16> %res
}

; COST-LABEL: smax.v8i16
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <8 x i16> @llvm.smax.v8i16(<8 x i16> %v0, <8 x i16> %v1)

; CODE-LABEL: smax.v8i16
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.8h, v{{.*}}.8h, v{{.*}}.8h
; CODE-NEXT:   ret

declare <8 x i16> @llvm.smax.v8i16(<8 x i16>, <8 x i16>)
define <8 x i16> @smax.v8i16(<8 x i16> %v0, <8 x i16> %v1) {
  %res = call <8 x i16> @llvm.smax.v8i16(<8 x i16> %v0, <8 x i16> %v1)
  ret <8 x i16> %res
}

; COST-LABEL: smax.v2i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <2 x i32> @llvm.smax.v2i32(<2 x i32> %v0, <2 x i32> %v1)

; CODE-LABEL: smax.v2i32
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.2s, v{{.*}}.2s, v{{.*}}.2s
; CODE-NEXT:   ret

declare <2 x i32> @llvm.smax.v2i32(<2 x i32>, <2 x i32>)
define <2 x i32> @smax.v2i32(<2 x i32> %v0, <2 x i32> %v1) {
  %res = call <2 x i32> @llvm.smax.v2i32(<2 x i32> %v0, <2 x i32> %v1)
  ret <2 x i32> %res
}

; COST-LABEL: smax.v4i32
; COST-NEXT: Cost Model: Found an estimated cost of 1 for instruction:   %res = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %v0, <4 x i32> %v1)

; CODE-LABEL: smax.v4i32
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <4 x i32> @llvm.smax.v4i32(<4 x i32>, <4 x i32>)
define <4 x i32> @smax.v4i32(<4 x i32> %v0, <4 x i32> %v1) {
  %res = call <4 x i32> @llvm.smax.v4i32(<4 x i32> %v0, <4 x i32> %v1)
  ret <4 x i32> %res
}

; COST-LABEL: smax.v8i32
; COST-NEXT: Cost Model: Found an estimated cost of 2 for instruction:   %res = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %v0, <8 x i32> %v1)

; CODE-LABEL: smax.v8i32
; CODE:       bb.0
; CODE-NEXT:   smax v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   smax v{{.*}}.4s, v{{.*}}.4s, v{{.*}}.4s
; CODE-NEXT:   ret

declare <8 x i32> @llvm.smax.v8i32(<8 x i32>, <8 x i32>)
define <8 x i32> @smax.v8i32(<8 x i32> %v0, <8 x i32> %v1) {
  %res = call <8 x i32> @llvm.smax.v8i32(<8 x i32> %v0, <8 x i32> %v1)
  ret <8 x i32> %res
}

; COST-LABEL: smax.v2i64
; COST-NEXT: Cost Model: Found an estimated cost of 6 for instruction:   %res = call <2 x i64> @llvm.smax.v2i64(<2 x i64> %v0, <2 x i64> %v1)

; CODE-LABEL: smax.v2i64
; CODE:       bb.0
; CODE:        csel
; CODE:        csel

declare <2 x i64> @llvm.smax.v2i64(<2 x i64>, <2 x i64>)
define <2 x i64> @smax.v2i64(<2 x i64> %v0, <2 x i64> %v1) {
  %res = call <2 x i64> @llvm.smax.v2i64(<2 x i64> %v0, <2 x i64> %v1)
  ret <2 x i64> %res
}
