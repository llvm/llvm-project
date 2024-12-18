; RUN: llc -mtriple=hexagon -hexagon-hvx-widen=16 < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: q[[Q0:[0-3]]] = vsetq(r{{[0-9]+}})
; CHECK: if (q[[Q0]]) vmem({{.*}}) = v
define void @f0(ptr %a0) #0 {
  %v0 = load <32 x i8>, ptr %a0, align 128
  %v1 = insertelement <32 x i8> undef, i8 1, i32 0
  %v2 = shufflevector <32 x i8> %v1, <32 x i8> undef, <32 x i32> zeroinitializer
  %v3 = add <32 x i8> %v0, %v2
  store <32 x i8> %v3, ptr %a0, align 128
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv65" "target-features"="+hvx,+hvx-length128b" }

