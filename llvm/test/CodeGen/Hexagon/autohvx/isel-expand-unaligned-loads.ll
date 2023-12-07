; RUN: llc -march=hexagon -disable-packetizer -hexagon-align-loads < %s | FileCheck %s

; CHECK-LABEL: test_00:
; CHECK-DAG: v[[V00:[0-9]+]] = vmem(r[[B00:[0-9]+]]+#0)
; CHECK-DAG: v[[V01:[0-9]+]] = vmem(r[[B00]]+#1)
; CHECK: valign(v[[V01]],v[[V00]],r[[B00]])
define void @test_00(ptr %p, ptr %q) #0 {
  %v0 = load <64 x i8>, ptr %p, align 1
  store <64 x i8> %v0, ptr %q, align 1
  ret void
}

; CHECK-LABEL: test_01:
; CHECK-DAG: v[[V10:[0-9]+]] = vmem(r[[B01:[0-9]+]]+#0)
; CHECK-DAG: v[[V11:[0-9]+]] = vmem(r[[B01]]+#1)
; CHECK-DAG: v[[V12:[0-9]+]] = vmem(r[[B01]]+#2)
; CHECK: }
; CHECK-DAG: valign(v[[V11]],v[[V10]],r[[B01]])
; CHECK-DAG: valign(v[[V12]],v[[V11]],r[[B01]])
define void @test_01(ptr %p, ptr %q) #0 {
  %v0 = load <128 x i8>, ptr %p, align 1
  store <128 x i8> %v0, ptr %q, align 1
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
