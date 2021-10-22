; RUN: not --crash llc -mtriple=riscv32 -mattr=+m,+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs < %s
; RUN: not --crash llc -mtriple=riscv64 -mattr=+m,+experimental-v,+experimental-zfh,+f,+d -verify-machineinstrs < %s

; Check that we are able to legalize scalable-vector stores that require widening.

; FIXME: LLVM can't yet widen scalable-vector stores.

define void @store_nxv3i8(<vscale x 3 x i8> %val, <vscale x 3 x i8>* %ptr) {
  store <vscale x 3 x i8> %val, <vscale x 3 x i8>* %ptr
  ret void
}

define void @store_nxv7f64(<vscale x 7 x double> %val, <vscale x 7 x double>* %ptr) {
  store <vscale x 7 x double> %val, <vscale x 7 x double>* %ptr
  ret void
}
