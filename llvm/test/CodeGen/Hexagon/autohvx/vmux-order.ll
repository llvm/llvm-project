; RUN: llc -march=hexagon < %s | FileCheck %s

; Check for the correct order of vmux operands: the vcmp.eq sets predicate
; bits for 0s in the mask.
; 
; CHECK: vmux(q{{[0-3]+}},v1,v0)

define <16 x i32> @fred(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %p = shufflevector <16 x i32> %v0, <16 x i32> %v1, <16 x i32> <i32 0,i32 17,i32 2,i32 19,i32 4,i32 21,i32 6,i32 23,i32 8,i32 25,i32 10,i32 27,i32 12,i32 29,i32 14,i32 31>
  ret <16 x i32> %p
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
