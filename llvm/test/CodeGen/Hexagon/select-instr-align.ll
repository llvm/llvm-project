; RUN: llc -march=hexagon -hexagon-align-loads=0 < %s | FileCheck %s

; CHECK-LABEL: aligned_load:
; CHECK: = vmem({{.*}})
define <16 x i32> @aligned_load(<16 x i32>* %p, <16 x i32> %a) #0 {
  %v = load <16 x i32>, <16 x i32>* %p, align 64
  ret <16 x i32> %v
}

; CHECK-LABEL: aligned_store:
; CHECK: vmem({{.*}}) =
define void @aligned_store(<16 x i32>* %p, <16 x i32> %a) #0 {
  store <16 x i32> %a, <16 x i32>* %p, align 64
  ret void
}

; CHECK-LABEL: unaligned_load:
; CHECK: = vmemu({{.*}})
define <16 x i32> @unaligned_load(<16 x i32>* %p, <16 x i32> %a) #0 {
  %v = load <16 x i32>, <16 x i32>* %p, align 32
  ret <16 x i32> %v
}

; CHECK-LABEL: unaligned_store:
; CHECK: vmemu({{.*}}) =
define void @unaligned_store(<16 x i32>* %p, <16 x i32> %a) #0 {
  store <16 x i32> %a, <16 x i32>* %p, align 32
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
