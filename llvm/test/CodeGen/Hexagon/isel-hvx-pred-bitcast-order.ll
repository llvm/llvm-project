; RUN: llc -march=hexagon < %s | FileCheck %s
;
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

define i64 @foo(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %v0 = icmp ugt <64 x i16> %a0, %a1
  %v1 = bitcast <64 x i1> %v0 to i64
  ret i64 %v1
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv66" "target-features"="+hvx,+hvx-length128b,-packets" }

