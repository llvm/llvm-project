; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target triple = "aarch64-linux-gnu"

define <vscale x 2 x i32> @ok_gather(<vscale x 2 x i1> %pg, ptr %base, <vscale x 2 x i64> %offsets) {
  %r = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg, ptr %base, <vscale x 2 x i64> %offsets)
  ret <vscale x 2 x i32> %r
}

define <vscale x 4 x i32> @ok_gather_packed(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i32> %offsets) {
  %r = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i32(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i32> %offsets)
  ret <vscale x 4 x i32> %r
}

; Narrower-than-Z scalable offsets are legal: the legalizer widens them.
define <vscale x 2 x i32> @ok_gather_narrow_offsets(<vscale x 2 x i1> %pg, ptr %base, <vscale x 2 x i32> %offsets) {
  %r = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv2i32(<vscale x 2 x i1> %pg, ptr %base, <vscale x 2 x i32> %offsets)
  ret <vscale x 2 x i32> %r
}

; Exactly the six calls below should be rejected.
; CHECK-COUNT-6: SVE gather/scatter offsets must be a scalable vector no wider than a single Z register
; CHECK-NOT: SVE gather/scatter offsets must be a scalable vector no wider than a single Z register

define <vscale x 4 x i32> @oversized_gather(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets) {
  %r = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.nxv4i32(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets)
  ret <vscale x 4 x i32> %r
}

define void @oversized_scatter(<vscale x 4 x i32> %value, <vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets) {
  call void @llvm.aarch64.sve.st1.scatter.nxv4i32(<vscale x 4 x i32> %value, <vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets)
  ret void
}

define <vscale x 8 x i32> @oversized_packed_gather(<vscale x 8 x i1> %pg, ptr %base, <vscale x 8 x i32> %offsets) {
  %r = call <vscale x 8 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv8i32(<vscale x 8 x i1> %pg, ptr %base, <vscale x 8 x i32> %offsets)
  ret <vscale x 8 x i32> %r
}

define void @oversized_prefetch(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets) {
  call void @llvm.aarch64.sve.prfd.gather.index.nxv4i64.p0(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i64> %offsets, i32 0)
  ret void
}

define <8 x i32> @oversized_fixed_offsets(<8 x i1> %pg, ptr %base, <8 x i32> %offsets) {
  %r = call <8 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.v8i32(<8 x i1> %pg, ptr %base, <8 x i32> %offsets)
  ret <8 x i32> %r
}

; Fixed-width offsets are rejected however narrow.
define <2 x i32> @fixed_offsets(<2 x i1> %pg, ptr %base, <2 x i64> %offsets) {
  %r = call <2 x i32> @llvm.aarch64.sve.ld1.gather.v2i32(<2 x i1> %pg, ptr %base, <2 x i64> %offsets)
  ret <2 x i32> %r
}
