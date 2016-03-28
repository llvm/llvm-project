; RUN: llc -march=hexagon -enable-hexagon-hvx < %s | FileCheck %s
; CHECK-LABEL: aligned_load:
; CHECK: = vmem({{.*}})
; CHECK-LABEL: aligned_store:
; CHECK: vmem({{.*}}) =
; CHECK-LABEL: unaligned_load:
; CHECK: = vmemu({{.*}})
; CHECK-LABEL: unaligned_store:
; CHECK: vmemu({{.*}}) =

define <16 x i32> @aligned_load(<16 x i32>* %p, <16 x i32> %a) {
  %v = load <16 x i32>, <16 x i32>* %p, align 64
  ret <16 x i32> %v
}

define void @aligned_store(<16 x i32>* %p, <16 x i32> %a) {
  store <16 x i32> %a, <16 x i32>* %p, align 64
  ret void
}

define <16 x i32> @unaligned_load(<16 x i32>* %p, <16 x i32> %a) {
  %v = load <16 x i32>, <16 x i32>* %p, align 32
  ret <16 x i32> %v
}

define void @unaligned_store(<16 x i32>* %p, <16 x i32> %a) {
  store <16 x i32> %a, <16 x i32>* %p, align 32
  ret void
}


