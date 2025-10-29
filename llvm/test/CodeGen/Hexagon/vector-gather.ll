; REQUIRES: hexagon-registered-target
; RUN: llc -march=hexagon -mcpu=hexagonv73 -mattr=+hvxv73,+hvx-length128b < %s | FileCheck %s

target triple = "hexagon"

@VTCM_SCATTER16_ADDRESS = dso_local global i32 0, align 4
@region_len = dso_local global i32 16383, align 4

; CHECK: [[ADR:r[0-9]+]] = memw(gp+#VTCM_SCATTER16_ADDRESS)
; CHECK: vtmp.h = vgather([[ADR]],m0,v0.h).h
; CHECK: vmem(r0+#0) = vtmp.new

define dso_local void @vector_gather_16(ptr noundef %vgather, <32 x i32> noundef %offsets) #0 {
entry:
  %vgather.addr = alloca ptr, align 4
  %offsets.addr = alloca <32 x i32>, align 128
  store ptr %vgather, ptr %vgather.addr, align 4
  store <32 x i32> %offsets, ptr %offsets.addr, align 128
  %0 = load ptr, ptr %vgather.addr, align 4
  %1 = load i32, ptr @VTCM_SCATTER16_ADDRESS, align 4
  %2 = load i32, ptr @region_len, align 4
  %3 = load <32 x i32>, ptr %offsets.addr, align 128
  call void @llvm.hexagon.V6.vgathermh.128B(ptr %0, i32 %1, i32 %2, <32 x i32> %3)
  ret void
}

declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32)
