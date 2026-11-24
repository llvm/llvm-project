; RUN: not --crash llc %s -mtriple=aarch64 -mattr=+sve -o /dev/null

; This should fail to select with only +sve.
define <vscale x 4 x half> @expand(ptr %p, <vscale x 4 x i1> %m) {
  %r = call <vscale x 4 x half> @llvm.masked.expandload.nxv4f16(ptr %p, <vscale x 4 x i1> %m, <vscale x 4 x half> poison)
  ret <vscale x 4 x half> %r
}
