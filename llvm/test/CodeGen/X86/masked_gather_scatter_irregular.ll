; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=znver4 | FileCheck %s

; Instruction counts for gathers and scatters whose vector length is not a power
; of two. X86TTIImpl::getGSVectorCost prices these by the count emitted here, so
; the two must agree.
;
; A v24 operation legalizes into four parts of eight whose last part holds no
; live lane. A gather folds it away. A variable-mask scatter keeps it as a store
; under a zeroed mask, and folds it only once the mask is known all-true.

define <24 x i32> @gather_v24i32(<24 x ptr> %ptrs, <24 x i1> %mask) {
; CHECK-LABEL: gather_v24i32:
; CHECK-COUNT-3: vpgatherqd
; CHECK-NOT: vpgatherqd
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> %mask, <24 x i32> poison)
  ret <24 x i32> %v
}

define void @scatter_v24i32(<24 x i32> %val, <24 x ptr> %ptrs, <24 x i1> %mask) {
; CHECK-LABEL: scatter_v24i32:
; CHECK-COUNT-4: vpscatterqd
; CHECK-NOT: vpscatterqd
  call void @llvm.masked.scatter.v24i32.v24p0(<24 x i32> %val, <24 x ptr> %ptrs, i32 4, <24 x i1> %mask)
  ret void
}

define <24 x i32> @gather_v24i32_allones(<24 x ptr> %ptrs) {
; CHECK-LABEL: gather_v24i32_allones:
; CHECK-COUNT-3: vpgatherqd
; CHECK-NOT: vpgatherqd
  %v = call <24 x i32> @llvm.masked.gather.v24i32.v24p0(<24 x ptr> %ptrs, i32 4, <24 x i1> splat(i1 true), <24 x i32> poison)
  ret <24 x i32> %v
}

define void @scatter_v24i32_allones(<24 x i32> %val, <24 x ptr> %ptrs) {
; CHECK-LABEL: scatter_v24i32_allones:
; CHECK-COUNT-3: vpscatterqd
; CHECK-NOT: vpscatterqd
  call void @llvm.masked.scatter.v24i32.v24p0(<24 x i32> %val, <24 x ptr> %ptrs, i32 4, <24 x i1> splat(i1 true))
  ret void
}
