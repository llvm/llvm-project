; RUN: opt -S -passes=instsimplify < %s | FileCheck %s
; RUN: verify-uselistorder %s
; PR18319

define <4 x i16> @function() {
  %s = select <4 x i1> <i1 undef, i1 undef, i1 false, i1 true>, <4 x i16> <i16 undef, i16 2, i16 3, i16 4>, <4 x i16> <i16 -1, i16 -2, i16 -3, i16 -4>
; CHECK: <i16 undef, i16 -2, i16 -3, i16 4>
  ret <4 x i16> %s
}
