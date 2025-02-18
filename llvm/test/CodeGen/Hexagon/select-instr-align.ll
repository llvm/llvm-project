; RUN: llc -mtriple=hexagon -hexagon-align-loads=0 < %s | FileCheck %s

; CHECK-LABEL: aligned_load:
; CHECK: = vmem({{.*}})
define <16 x i32> @aligned_load(ptr %p, <16 x i32> %a) #0 {
  %v = load <16 x i32>, ptr %p, align 64
  ret <16 x i32> %v
}

; CHECK-LABEL: aligned_store:
; CHECK: vmem({{.*}}) =
define void @aligned_store(ptr %p, <16 x i32> %a) #0 {
  store <16 x i32> %a, ptr %p, align 64
  ret void
}

; CHECK-LABEL: unaligned_load:
; CHECK: = vmemu({{.*}})
define <16 x i32> @unaligned_load(ptr %p, <16 x i32> %a) #0 {
  %v = load <16 x i32>, ptr %p, align 32
  ret <16 x i32> %v
}

; CHECK-LABEL: unaligned_store:
; CHECK: vmemu({{.*}}) =
define void @unaligned_store(ptr %p, <16 x i32> %a) #0 {
  store <16 x i32> %a, ptr %p, align 32
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length64b" }
