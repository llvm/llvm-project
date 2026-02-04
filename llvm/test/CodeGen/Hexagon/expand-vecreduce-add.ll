; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target triple = "hexagon"

define i32 @add_v32i32(<32 x i32> %vec) #0 {
; CHECK-LABEL: add_v32i32:
; CHECK: {
; CHECK: [[R0:v[0-9]+]] = valign([[_:v[0-9]+]],v0,{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R1:v[0-9]+]].w = vadd(v0.w,[[R0]].w)
; CHECK: }
; CHECK: {
; CHECK: [[R2:v[0-9]+]] = valign([[_:v[0-9]+]],[[R1]],{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R3:v[0-9]+]].w = vadd([[R1]].w,[[R2]].w)
; CHECK: }
; CHECK: {
; CHECK: [[R4:v[0-9]+]] = valign([[_:v[0-9]+]],[[R3]],{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R5:v[0-9]+]].w = vadd([[R3]].w,[[R4]].w)
; CHECK: }
; CHECK: {
; CHECK: [[R6:v[0-9]+]] = valign([[_:v[0-9]+]],[[R5]],{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R7:v[0-9]+]].w = vadd([[R5]].w,[[R6]].w)
; CHECK: }
; CHECK: {
; CHECK: [[R8:v[0-9]+]] = valign([[_:v[0-9]+]],[[R7]],{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R9:v[0-9]+]].w = vadd([[R7]].w,[[R8]].w)
; CHECK: }
; CHECK: {
; CHECK: r0 = vextract([[R9]],{{.+}})
; CHECK: }
entry:
  %r = call i32 @llvm.vector.reduce.add.v32i32(<32 x i32> %vec)
  ret i32 %r
}

define i32 @add_v16i32(<16 x i32> %vec) #0 {
; CHECK-LABEL: add_v16i32:
; CHECK: {
; CHECK: [[R0:v[0-9]+]] = valign([[_:v[0-9]+]],v0,{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R1:v[0-9]+]].w = vadd(v0.w,[[R0]].w)
; CHECK: }
; CHECK: {
; CHECK: [[R2:v[0-9]+]] = valign([[_:v[0-9]+]],[[R1]],{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R3:v[0-9]+]].w = vadd([[R1]].w,[[R2]].w)
; CHECK: }
; CHECK: {
; CHECK: [[R4:v[0-9]+]] = valign([[_:v[0-9]+]],[[R3]],{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R5:v[0-9]+]].w = vadd([[R3]].w,[[R4]].w)
; CHECK: }
; CHECK: {
; CHECK: [[R6:v[0-9]+]] = valign([[_:v[0-9]+]],[[R5]],{{.+}})
; CHECK: }
; CHECK: {
; CHECK: [[R7:v[0-9]+]].w = vadd([[R5]].w,[[R6]].w)
; CHECK: }
; CHECK: {
; CHECK: r0 = vextract([[R7]],{{.+}})
; CHECK: }
entry:
  %r = call i32 @llvm.vector.reduce.add.v16i32(<16 x i32> %vec)
  ret i32 %r
}

define i32 @add_v8i32(<8 x i32> %vec) #0 {
; CHECK-LABEL: add_v8i32:
; CHECK: {
; CHECK: r[[RS1:[0-9]+:[0-9]+]] = vaddw(r1:0,r5:4)
; CHECK: r[[R6:[0-9]+:[0-9]+]] = memd(r29+#0)
; CHECK: }
; CHECK: {
; CHECK: r[[RS2:[0-9]+:[0-9]+]] = vaddw(r3:2,r[[R6]])
; CHECK: }
; CHECK: {
; CHECK: r[[RS3:[0-9]+:[0-9]+]] = vaddw(r[[RS1]],r[[RS2]])
; CHECK: }
; CHECK: {
;; TODO: combine and double register add can be optimized to single register add.
; CHECK: r[[RS4:[0-9]+:[0-9]+]] = combine(#0,r{{[0-9]+}})
; CHECK: }
; CHECK: {
; CHECK: r1:0 = vaddw(r[[RS3]],r[[RS4]])
entry:
  %r = call i32 @llvm.vector.reduce.add.v8i32(<8 x i32> %vec)
  ret i32 %r
}

define i32 @add_v64i32(<64 x i32> %vec) #0 {
; CHECK-LABEL: add_v64i32:
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
; CHECK: valign
; CHECK: vadd
entry:
  %r = call i32 @llvm.vector.reduce.add.v64i32(<64 x i32> %vec)
  ret i32 %r
}

;; Non-pow2 vectors are scalarized.

define i32 @add_v12i32(<12 x i32> %vec) #0 {
; CHECK-LABEL: add_v12i32:
; CHECK: [[RS0:r[0-9]+]] = add(r0,r1)
; CHECK: [[RS1:r[0-9]+]] += add([[RS0]],r{{[0-9]+}})
; CHECK: [[RS2:r[0-9]+]] += add([[RS1]],r{{[0-9]+}})
; CHECK: [[RS3:r[0-9]+]] += add([[RS2]],r{{[0-9]+}})
; CHECK: [[RS4:r[0-9]+]] += add([[RS3]],r{{[0-9]+}})
; CHECK: [[RS5:r[0-9]+]] += add([[RS4]],r{{[0-9]+}})
entry:
  %r = call i32 @llvm.vector.reduce.add.v12i32(<12 x i32> %vec)
  ret i32 %r
}

define i32 @add_v3i32(<3 x i32> %vec) #0 {
; CHECK-LABEL: add_v3i32:
; CHECK: r{{[0-9]+}} += add(r{{[0-9]+}},r{{[0-9]+}})
entry:
  %r = call i32 @llvm.vector.reduce.add.v3i32(<3 x i32> %vec)
  ret i32 %r
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv68" "target-features"="+hvx,+hvx-length128b" }
