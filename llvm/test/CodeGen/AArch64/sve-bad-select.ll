; RUN: not llc -mtriple=aarch64-linux-gnu -mattr=+sve < %s 2>&1 | FileCheck %s

define <vscale x 16 x i8> @badsel1_nxv16i8(<16 x i1> %p,
                                           <vscale x 16 x i8> %dst,
                                           <vscale x 16 x i8> %a) {
  %sel = select <16 x i1> %p, <vscale x 16 x i8> %a, <vscale x 16 x i8> %dst
  ret <vscale x 16 x i8> %sel
}

; CHECK: error: vector select requires selected vectors to have the same vector length as select condition
