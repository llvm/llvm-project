; RUN: llc -march=hexagon -hexagon-hvx-widen=32 < %s | FileCheck %s

; If the "rx = #N, vsetq(rx)" get reordered with the rest, update the test.

; v32i16 -> v32i8
; CHECK-LABEL: f0:
; CHECK: r[[R0:[0-9]+]] = #32
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]].b = vdeal(v[[V0]].b)
; CHECK: q[[Q0:[0-3]]] = vsetq(r[[R0]])
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V1]]
define void @f0(<32 x i16>* %a0, <32 x i8>* %a1) #0 {
  %v0 = load <32 x i16>, <32 x i16>* %a0, align 128
  %v1 = trunc <32 x i16> %v0 to <32 x i8>
  store <32 x i8> %v1, <32 x i8>* %a1, align 128
  ret void
}

; v32i32 -> v32i8
; CHECK-LABEL: f1:
; CHECK: r[[R0:[0-9]+]] = #32
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]].b = vdeale({{.*}},v[[V0]].b)
; CHECK: q[[Q0:[0-3]]] = vsetq(r[[R0]])
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V1]]
define void @f1(<32 x i32>* %a0, <32 x i8>* %a1) #0 {
  %v0 = load <32 x i32>, <32 x i32>* %a0, align 128
  %v1 = trunc <32 x i32> %v0 to <32 x i8>
  store <32 x i8> %v1, <32 x i8>* %a1, align 128
  ret void
}

; v64i16 -> v64i8
; CHECK-LABEL: f2:
; CHECK: r[[R0:[0-9]+]] = #64
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]].b = vdeal(v[[V0]].b)
; CHECK: q[[Q0:[0-3]]] = vsetq(r[[R0]])
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V1]]
define void @f2(<64 x i16>* %a0, <64 x i8>* %a1) #0 {
  %v0 = load <64 x i16>, <64 x i16>* %a0, align 128
  %v1 = trunc <64 x i16> %v0 to <64 x i8>
  store <64 x i8> %v1, <64 x i8>* %a1, align 128
  ret void
}

; v64i32 -> v64i8
; CHECK-LABEL: f3:
; CHECK-DAG: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK-DAG: v[[V1:[0-9]+]] = vmem(r0+#1)
; CHECK-DAG: q[[Q0:[0-3]]] = vsetq
; CHECK: v[[V2:[0-9]+]].b = vdeale(v[[V1]].b,v[[V0]].b)
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V2]]
define void @f3(<64 x i32>* %a0, <64 x i8>* %a1) #0 {
  %v0 = load <64 x i32>, <64 x i32>* %a0, align 128
  %v1 = trunc <64 x i32> %v0 to <64 x i8>
  store <64 x i8> %v1, <64 x i8>* %a1, align 128
  ret void
}

; v16i32 -> v16i16
; CHECK-LABEL: f4:
; CHECK: r[[R0:[0-9]+]] = #32
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]].h = vdeal(v[[V0]].h)
; CHECK: q[[Q0:[0-3]]] = vsetq(r[[R0]])
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V1]]
define void @f4(<16 x i32>* %a0, <16 x i16>* %a1) #0 {
  %v0 = load <16 x i32>, <16 x i32>* %a0, align 128
  %v1 = trunc <16 x i32> %v0 to <16 x i16>
  store <16 x i16> %v1, <16 x i16>* %a1, align 128
  ret void
}

; v32i32 -> v32i16
; CHECK-LABEL: f5:
; CHECK: r[[R0:[0-9]+]] = #64
; CHECK: v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK: v[[V1:[0-9]+]].h = vdeal(v[[V0]].h)
; CHECK: q[[Q0:[0-3]]] = vsetq(r[[R0]])
; CHECK: if (q[[Q0]]) vmem(r1+#0) = v[[V1]]
define void @f5(<32 x i32>* %a0, <32 x i16>* %a1) #0 {
  %v0 = load <32 x i32>, <32 x i32>* %a0, align 128
  %v1 = trunc <32 x i32> %v0 to <32 x i16>
  store <32 x i16> %v1, <32 x i16>* %a1, align 128
  ret void
}

; v8i32 -> v8i8
; CHECK-LABEL: f6:
; CHECK:     v[[V0:[0-9]+]] = vmem(r0+#0)
; CHECK:     v[[V1:[0-9]+]].b = vdeale({{.*}},v[[V0]].b)
; CHECK:     vmem(r[[R0:[0-9]+]]+#0) = v[[V1]]
; CHECK-DAG: r[[R1:[0-9]+]] = memw(r[[R0]]+#0)
; CHECK-DAG: r[[R2:[0-9]+]] = memw(r[[R0]]+#4)
; CHECK:     memd(r1+#0) = r[[R2]]:[[R1]]
define void @f6(<8 x i32>* %a0, <8 x i8>* %a1) #0 {
  %v0 = load <8 x i32>, <8 x i32>* %a0, align 128
  %v1 = trunc <8 x i32> %v0 to <8 x i8>
  store <8 x i8> %v1, <8 x i8>* %a1, align 128
  ret void
}


attributes #0 = { "target-cpu"="hexagonv65" "target-features"="+hvx,+hvx-length128b,-packets" }

