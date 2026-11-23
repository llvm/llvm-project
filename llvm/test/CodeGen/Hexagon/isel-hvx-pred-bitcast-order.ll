; RUN: llc -mtriple=hexagon < %s | FileCheck %s
;
; CHECK-LABEL: foo_128_2x

; Check that bitcast is computed corrected by fixing up a predicate register
; before compressing it.
; CHECK: v[[V1:[0-9]+]].b = vdeal(v{{[0-9]+}}.b)

; Check that the resulting register pair has the registers in the right order.

; CHECK: vdeal
; CHECK: vdeal
; CHECK: v[[V1:[0-9]+]]:[[V0:[0-9]+]] = vshuff
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: vmem(r[[RA:[0-9]+]]+#0) = v[[V0]]
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: r0 = memw(r1+#0)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: r1 = memw(r1+#4)
; CHECK-NEXT: }
; CHECK-NEXT: {
; CHECK-NEXT: r31:30 = dealloc_return(r30):raw
; CHECK-NEXT: }

define i64 @foo_128_2x(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %v0 = icmp ugt <64 x i16> %a0, %a1
  %v1 = bitcast <64 x i1> %v0 to i64
  ret i64 %v1
}

;; The remaining test cases only check the bitvector compression stage, which is
;; the only distinction.

; CHECK-LABEL: foo_128_4x
; CHECK: v[[V1:[0-9]+]].h = vdeal(v{{[0-9]+}}.h)
; CHECK: v{{[0-9]+}}.b = vdeal(v[[V1]].b)
; CHECK: vdeal
; CHECK: vdeal
; CHECK: vshuff
define i32 @foo_128_4x(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %v0 = icmp ugt <32 x i16> %a0, %a1
  %v1 = bitcast <32 x i1> %v0 to i32
  ret i32 %v1
}

; CHECK-LABEL: foo_64_2x
; CHECK: v{{[0-9]+}}.b = vdeal(v{{[0-9]+}}.b)
; CHECK: vdeal
; CHECK: vdeal
; CHECK: vshuff
define i32 @foo_64_2x(<32 x i16> %a0, <32 x i16> %a1) #1 {
  %v0 = icmp ugt <32 x i16> %a0, %a1
  %v1 = bitcast <32 x i1> %v0 to i32
  ret i32 %v1
}

; CHECK-LABEL: foo_64_4x
; CHECK: v[[V1:[0-9]+]].h = vdeal(v{{[0-9]+}}.h)
; CHECK: v{{[0-9]+}}.b = vdeal(v[[V1]].b)
; CHECK: vdeal
; CHECK: vdeal
; CHECK: vshuff
define i16 @foo_64_4x(<16 x i16> %a0, <16 x i16> %a1) #1 {
  %v0 = icmp ugt <16 x i16> %a0, %a1
  %v1 = bitcast <16 x i1> %v0 to i16
  ret i16 %v1
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv68" "target-features"="+hvx,+hvx-length128b,-packets" }
attributes #1 = { nounwind readnone "target-cpu"="hexagonv68" "target-features"="+hvx,+hvx-length64b,-packets" }

