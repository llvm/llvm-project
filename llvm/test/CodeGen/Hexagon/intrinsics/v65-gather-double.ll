; RUN: llc -mv65 -mattr=+hvxv65,hvx-length128b -mtriple=hexagon -O2 < %s | FileCheck %s

; CHECK-LABEL: V6_vgathermw_128B
; CHECK: vtmp.w = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.w).w
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermh_128B
; CHECK: vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.h).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermhw_128B
; CHECK: vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermwq_128B
; CHECK: if (q{{[0-3]+}}) vtmp.w = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.w).w
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermhq_128B
; CHECK: if (q{{[0-3]+}}) vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.h).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermhwq_128B
; CHECK: if (q{{[0-3]+}}) vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new

declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32)

declare void @llvm.hexagon.V6.vgathermw.128B(ptr, i32, i32, <32 x i32>)
define void @V6_vgathermw_128B(ptr %a, i32 %b, i32 %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vgathermw.128B(ptr %a, i32 %b, i32 %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vgathermh.128B(ptr, i32, i32, <32 x i32>)
define void @V6_vgathermh_128B(ptr %a, i32 %b, i32 %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vgathermh.128B(ptr %a, i32 %b, i32 %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vgathermhw.128B(ptr, i32, i32, <64 x i32>)
define void @V6_vgathermhw_128B(ptr %a, i32 %b, i32 %c, <64 x i32> %d) {
  call void @llvm.hexagon.V6.vgathermhw.128B(ptr %a, i32 %b, i32 %c, <64 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vgathermwq.128B(ptr, <128 x i1>, i32, i32, <32 x i32>)
define void @V6_vgathermwq_128B(ptr %a, <32 x i32> %b, i32 %c, i32 %d, <32 x i32> %e) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %b, i32 -1)
  call void @llvm.hexagon.V6.vgathermwq.128B(ptr %a, <128 x i1> %1, i32 %c, i32 %d, <32 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vgathermhq.128B(ptr, <128 x i1>, i32, i32, <32 x i32>)
define void @V6_vgathermhq_128B(ptr %a, <32 x i32> %b, i32 %c, i32 %d, <32 x i32> %e) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %b, i32 -1)
  call void @llvm.hexagon.V6.vgathermhq.128B(ptr %a, <128 x i1> %1, i32 %c, i32 %d, <32 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vgathermhwq.128B(ptr, <128 x i1>, i32, i32, <64 x i32>)
define void @V6_vgathermhwq_128B(ptr %a, <32 x i32> %b, i32 %c, i32 %d, <64 x i32> %e) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %b, i32 -1)
  call void @llvm.hexagon.V6.vgathermhwq.128B(ptr %a, <128 x i1> %1, i32 %c, i32 %d, <64 x i32> %e)
  ret void
}

