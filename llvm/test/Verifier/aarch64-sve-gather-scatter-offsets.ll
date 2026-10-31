; RUN: not llvm-as < %s 2>&1 | FileCheck %s

target triple = "aarch64-linux-gnu"

declare <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1>, ptr, <vscale x 2 x i64>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i32(<vscale x 4 x i1>, ptr, <vscale x 4 x i32>)
declare <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.nxv4i32(<vscale x 4 x i1>, ptr, <vscale x 4 x i64>)
declare void @llvm.aarch64.sve.st1.scatter.nxv4i32(<vscale x 4 x i32>, <vscale x 4 x i1>, ptr, <vscale x 4 x i64>)
declare <vscale x 8 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv8i32(<vscale x 8 x i1>, ptr, <vscale x 8 x i32>)

define <vscale x 2 x i32> @ok_gather(<vscale x 2 x i1> %pg, ptr %base, <vscale x 2 x i64> %offsets) {
  %r = call <vscale x 2 x i32> @llvm.aarch64.sve.ld1.gather.nxv2i32(<vscale x 2 x i1> %pg, ptr %base, <vscale x 2 x i64> %offsets)
  ret <vscale x 2 x i32> %r
}

define <vscale x 4 x i32> @ok_gather_packed(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i32> %offsets) {
  %r = call <vscale x 4 x i32> @llvm.aarch64.sve.ld1.gather.sxtw.nxv4i32(<vscale x 4 x i1> %pg, ptr %base, <vscale x 4 x i32> %offsets)
  ret <vscale x 4 x i32> %r
}

; Exactly the three calls below should be rejected.
; CHECK-COUNT-3: SVE gather/scatter offsets must fit in a single Z register

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
