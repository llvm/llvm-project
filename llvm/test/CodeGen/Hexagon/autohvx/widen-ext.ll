; RUN: llc -march=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

; v32i8 -> v32i16
; CHECK-LABEL: f0:
; CHECK: r[[R0:[0-9]+]] = #64
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]]:[[V2:[0-9]+]].h = vunpack(v[[V0]].b)
; CHECK: q[[Q0:[0-3]]] = vsetq(r[[R0]])
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V2]]
define void @f0(<32 x i8>* %a0, <32 x i16>* %a1) #0 {
  %v0 = load <32 x i8>, <32 x i8>* %a0, align 128
  %v1 = sext <32 x i8> %v0 to <32 x i16>
  store <32 x i16> %v1, <32 x i16>* %a1, align 128
  ret void
}

; v32i8 -> v32i32
; CHECK-LABEL: f1:
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]]:[[V2:[0-9]+]].h = vunpack(v[[V0]].b)
; CHECK: v[[V3:[0-9]+]]:[[V4:[0-9]+]].w = vunpack(v[[V2]].h)
; CHECK: vmem(r1+#0) = v[[V4]]
define void @f1(<32 x i8>* %a0, <32 x i32>* %a1) #0 {
  %v0 = load <32 x i8>, <32 x i8>* %a0, align 128
  %v1 = sext <32 x i8> %v0 to <32 x i32>
  store <32 x i32> %v1, <32 x i32>* %a1, align 128
  ret void
}

; v64i8 -> v64i16
; CHECK-LABEL: f2:
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]]:[[V2:[0-9]+]].h = vunpack(v[[V0]].b)
; CHECK: vmem(r1+#0) = v[[V2]]
define void @f2(<64 x i8>* %a0, <64 x i16>* %a1) #0 {
  %v0 = load <64 x i8>, <64 x i8>* %a0, align 128
  %v1 = sext <64 x i8> %v0 to <64 x i16>
  store <64 x i16> %v1, <64 x i16>* %a1, align 128
  ret void
}

; v64i8 -> v64i32
; CHECK-LABEL: f3:
; CHECK:     v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK:     v[[V1:[0-9]+]]:[[V2:[0-9]+]].h = vunpack(v[[V0]].b)
; CHECK:     v[[V3:[0-9]+]]:[[V4:[0-9]+]].w = vunpack(v[[V2]].h)
; CHECK-DAG: vmem(r1+#0) = v[[V4]]
; CHECK-DAG: vmem(r1+#1) = v[[V3]]
define void @f3(<64 x i8>* %a0, <64 x i32>* %a1) #0 {
  %v0 = load <64 x i8>, <64 x i8>* %a0, align 128
  %v1 = sext <64 x i8> %v0 to <64 x i32>
  store <64 x i32> %v1, <64 x i32>* %a1, align 128
  ret void
}

; v16i16 -> v16i32
; CHECK-LABEL: f4:
; CHECK: r[[R0:[0-9]+]] = #64
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]]:[[V2:[0-9]+]].w = vunpack(v[[V0]].h)
; CHECK: q[[Q0:[0-3]]] = vsetq(r[[R0]])
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V2]]
define void @f4(<16 x i16>* %a0, <16 x i32>* %a1) #0 {
  %v0 = load <16 x i16>, <16 x i16>* %a0, align 128
  %v1 = sext <16 x i16> %v0 to <16 x i32>
  store <16 x i32> %v1, <16 x i32>* %a1, align 128
  ret void
}

; v32i16 -> v32i32
; CHECK-LABEL: f5:
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]]:[[V2:[0-9]+]].w = vunpack(v[[V0]].h)
; CHECK: vmem(r1+#0) = v[[V2]]
define void @f5(<32 x i16>* %a0, <32 x i32>* %a1) #0 {
  %v0 = load <32 x i16>, <32 x i16>* %a0, align 128
  %v1 = sext <32 x i16> %v0 to <32 x i32>
  store <32 x i32> %v1, <32 x i32>* %a1, align 128
  ret void
}

; v8i8 -> v8i32
; CHECK-LABEL: f6:
; CHECK:     r[[R0:[0-9]+]]:[[R1:[0-9]+]] = memd(r0+#0)
; CHECK-DAG: v[[V0:[0-9]+]].w = vinsert(r[[R0]])
; CHECK-DAG: v[[V0]].w = vinsert(r[[R1]])
; CHECK-DAG: q[[Q0:[0-3]]] = vsetq
; CHECK:     v[[V1:[0-9]+]]:[[V2:[0-9]+]].h = vunpack(v[[V0]].b)
; CHECK:     v[[V3:[0-9]+]]:[[V4:[0-9]+]].w = vunpack(v[[V2]].h)
; CHECK:     if (q[[Q0]]) vmem(r1+#0) = v[[V4]]
define void @f6(<8 x i8>* %a0, <8 x i32>* %a1) #0 {
  %v0 = load <8 x i8>, <8 x i8>* %a0, align 128
  %v1 = sext <8 x i8> %v0 to <8 x i32>
  store <8 x i32> %v1, <8 x i32>* %a1, align 128
  ret void
}

attributes #0 = { "target-cpu"="hexagonv65" "target-features"="+hvx,+hvx-length128b,-packets" }

