; RUN: llc -march=hexagon -mv68 -mattr=+hmx < %s | FileCheck %s

; Make sure that the M8_mxmem2_bias and store are not packetized together.
; M8_mxmem2_bias is a pure SLOT0 instruction and has a A_RESTRICT_NOSLOT1_STORE
; constraint.
; S2_storeri_io can go in either SLOT0 or SLOT1.
; This implies we can not packetize these two instructions together.

; CHECK-LABEL: foo1
; CHECK: mxmem2
; CHECK-NEXT: }
; CHECK: memw

target triple = "hexagon"
declare void @llvm.hexagon.M8.mxmem2.bias(ptr) #0

define void @foo1(ptr %a, ptr %b) local_unnamed_addr {
entry:
  call void @llvm.hexagon.M8.mxmem2.bias(ptr %b)
  store i32 0, ptr %a, align 4
  ret void
}

; CHECK-LABEL: foo2
; CHECK: memw
; CHECK-NEXT: }
; CHECK: mxmem

define void @foo2(ptr %a, ptr %b) local_unnamed_addr {
entry:
  store i32 0, ptr %a, align 4
  call void @llvm.hexagon.M8.mxmem2.bias(ptr %b)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(write) }
