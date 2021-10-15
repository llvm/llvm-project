; RUN: not --crash llc -mtriple=riscv32 -mattr=+m,+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs < %s
; RUN: not --crash llc -mtriple=riscv64 -mattr=+m,+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs < %s

; Check that we are able to legalize scalable-vector loads that require widening.

; FIXME: LLVM can't yet widen scalable-vector loads.

define <vscale x 3 x i8> @load_nxv3i8(<vscale x 3 x i8>* %ptr) {
  %v = load <vscale x 3 x i8>, <vscale x 3 x i8>* %ptr
  ret <vscale x 3 x i8> %v
}

define <vscale x 5 x half> @load_nxv5f16(<vscale x 5 x half>* %ptr) {
  %v = load <vscale x 5 x half>, <vscale x 5 x half>* %ptr
  ret <vscale x 5 x half> %v
}
