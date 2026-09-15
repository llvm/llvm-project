; RUN: opt -data-layout="e-p:64:64-i64:64-n32:64-S128" -passes='sroa<aggregate-to-vector>' -S %s | FileCheck %s --check-prefixes=COMMON,INHERIT
; RUN: opt -data-layout="e-p:64:64-i64:64-n32:64-S128" -passes='sroa<aggregate-to-vector>,infer-alignment' -S %s | FileCheck %s --check-prefixes=COMMON,STACK128
; RUN: opt -data-layout="e-p:64:64-i64:64-n32:64-S64" -passes='sroa<aggregate-to-vector>' -S %s | FileCheck %s --check-prefixes=COMMON,INHERIT
; RUN: opt -data-layout="e-p:64:64-i64:64-n32:64-S64" -passes='sroa<aggregate-to-vector>,infer-alignment' -S %s | FileCheck %s --check-prefixes=COMMON,STACK64
; RUN: opt -data-layout="e-p:64:64-i64:64-n32:64" -passes='sroa<aggregate-to-vector>' -S %s | FileCheck %s --check-prefixes=COMMON,INHERIT
; RUN: opt -data-layout="e-p:64:64-i64:64-n32:64" -passes='sroa<aggregate-to-vector>,infer-alignment' -S %s | FileCheck %s --check-prefixes=COMMON,UNSPECIFIED

; SROA inherits the original alloca alignment, including at partition offsets.
; InferAlignment can raise it later when the natural stack alignment permits.
target datalayout = "e-p:64:64-i64:64-n32:64-S128"

%ptr5 = type { ptr, ptr, ptr, ptr, ptr }
%ptr2 = type { ptr, ptr }

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)

define void @natural_stack_alignment(ptr %src) {
; COMMON-LABEL: define void @natural_stack_alignment(
; COMMON-NEXT: entry:
; INHERIT-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 8
; STACK128-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 8
; STACK64-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 8
; UNSPECIFIED-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 64
; COMMON-NEXT: %slot.sroa.0.0.copyload = load volatile <5 x ptr>, ptr %src, align 8
; INHERIT-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 8
; STACK128-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 8
; STACK64-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 8
; UNSPECIFIED-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 64
; COMMON-NEXT: ret void
entry:
  %slot = alloca %ptr5, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %slot, ptr align 8 %src,
                                   i64 40, i1 true)
  ret void
}

define void @explicit_above_stack_alignment(ptr %src) {
; COMMON-LABEL: define void @explicit_above_stack_alignment(
; COMMON-NEXT: entry:
; INHERIT-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 32
; STACK128-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 32
; STACK64-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 32
; UNSPECIFIED-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 64
; COMMON-NEXT: %slot.sroa.0.0.copyload = load volatile <5 x ptr>, ptr %src, align 8
; INHERIT-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 32
; STACK128-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 32
; STACK64-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 32
; UNSPECIFIED-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 64
; COMMON-NEXT: ret void
entry:
  %slot = alloca %ptr5, align 32
  call void @llvm.memcpy.p0.p0.i64(ptr align 32 %slot, ptr align 8 %src,
                                   i64 40, i1 true)
  ret void
}

define void @explicit_preferred_alignment(ptr %src) {
; COMMON-LABEL: define void @explicit_preferred_alignment(
; COMMON-NEXT: entry:
; INHERIT-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 64
; STACK128-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 64
; STACK64-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 64
; UNSPECIFIED-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 64
; COMMON-NEXT: %slot.sroa.0.0.copyload = load volatile <5 x ptr>, ptr %src, align 8
; INHERIT-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 64
; STACK128-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 64
; STACK64-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 64
; UNSPECIFIED-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 64
; COMMON-NEXT: ret void
entry:
  %slot = alloca %ptr5, align 64
  call void @llvm.memcpy.p0.p0.i64(ptr align 64 %slot, ptr align 8 %src,
                                   i64 40, i1 true)
  ret void
}

define void @explicit_above_preferred_alignment(ptr %src) {
; COMMON-LABEL: define void @explicit_above_preferred_alignment(
; COMMON-NEXT: entry:
; INHERIT-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 128
; STACK128-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 128
; STACK64-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 128
; UNSPECIFIED-NEXT: %slot.sroa.0 = alloca <5 x ptr>, align 128
; COMMON-NEXT: %slot.sroa.0.0.copyload = load volatile <5 x ptr>, ptr %src, align 8
; INHERIT-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 128
; STACK128-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 128
; STACK64-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 128
; UNSPECIFIED-NEXT: store volatile <5 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 128
; COMMON-NEXT: ret void
entry:
  %slot = alloca %ptr5, align 128
  call void @llvm.memcpy.p0.p0.i64(ptr align 128 %slot, ptr align 8 %src,
                                   i64 40, i1 true)
  ret void
}

define void @promote_later_within_stack_alignment(ptr %src) {
; COMMON-LABEL: define void @promote_later_within_stack_alignment(
; COMMON-NEXT: entry:
; INHERIT-NEXT: %slot.sroa.0 = alloca <2 x ptr>, align 8
; STACK128-NEXT: %slot.sroa.0 = alloca <2 x ptr>, align 16
; STACK64-NEXT: %slot.sroa.0 = alloca <2 x ptr>, align 8
; UNSPECIFIED-NEXT: %slot.sroa.0 = alloca <2 x ptr>, align 16
; COMMON-NEXT: %slot.sroa.0.0.copyload = load volatile <2 x ptr>, ptr %src, align 8
; INHERIT-NEXT: store volatile <2 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 8
; STACK128-NEXT: store volatile <2 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 16
; STACK64-NEXT: store volatile <2 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 8
; UNSPECIFIED-NEXT: store volatile <2 x ptr> %slot.sroa.0.0.copyload, ptr %slot.sroa.0, align 16
; COMMON-NEXT: ret void
entry:
  %slot = alloca %ptr2, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %slot, ptr align 8 %src,
                                   i64 16, i1 true)
  ret void
}
