; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-DAG: v{{[0-9]+}} = vmem(r{{[0-9]+}}+#0)
; CHECK-DAG: v{{[0-9]+}} = vmem(r{{[0-9]+}}+#0)
; CHECK-DAG: vmem(r{{[0-9]+}}+#{{[0-1]}}) = v{{[0-9]+}}
; CHECK-DAG: vmem(r{{[0-9]+}}+#{{[0-1]}}) = v{{[0-9]+}}

target triple = "hexagon"

; Function Attrs: nounwind
define void @f0(<32 x i32> %a0, <32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, <32 x i32> %a4, <32 x i32> %a5, <32 x i32> %a6, <32 x i32> %a7, <32 x i32> %a8, <32 x i32> %a9, <32 x i32> %a10, <32 x i32> %a11, <32 x i32> %a12, <32 x i32> %a13, <32 x i32> %a14, <32 x i32> %a15, <32 x i32> %a16, <32 x i32> %a17) #0 {
b0:
  tail call void @f1(<32 x i32> %a1, <32 x i32> %a2, <32 x i32> %a3, <32 x i32> %a4, <32 x i32> %a5, <32 x i32> %a6, <32 x i32> %a7, <32 x i32> %a8, <32 x i32> %a9, <32 x i32> %a10, <32 x i32> %a11, <32 x i32> %a12, <32 x i32> %a13, <32 x i32> %a14, <32 x i32> %a15, <32 x i32> %a16, <32 x i32> %a17, <32 x i32> %a0) #0
  ret void
}

declare void @f1(<32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>, <32 x i32>) #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvxv60,+hvx-length128b,-packets" }
