; RUN: llc -mv65 -mattr=+hvxv65,hvx-length128b -march=hexagon -O2 < %s | FileCheck %s

; CHECK-LABEL: V6_vscattermw_128B
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.w).w = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermh_128B
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.h).h = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermw_add_128B
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.w).w += v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermh_add_128B
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.h).h += v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermwq_128B
; CHECK: if (q{{[0-3]}}) vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.w).w = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhq_128B
; CHECK: if (q{{[0-3]}}) vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.h).h = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhw_128B
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhw_add_128B
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h += v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhwq_128B
; CHECK: if (q{{[0-3]}}) vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h = v{{[0-9]+}}

declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32)

declare void @llvm.hexagon.V6.vscattermw.128B(i32, i32, <32 x i32>, <32 x i32>)
define void @V6_vscattermw_128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermw.128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermh.128B(i32, i32, <32 x i32>, <32 x i32>)
define void @V6_vscattermh_128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermh.128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermw.add.128B(i32, i32, <32 x i32>, <32 x i32>)
define void @V6_vscattermw_add_128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermw.add.128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermh.add.128B(i32, i32, <32 x i32>, <32 x i32>)
define void @V6_vscattermh_add_128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermh.add.128B(i32 %a, i32 %b, <32 x i32> %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermwq.128B(<128 x i1>, i32, i32, <32 x i32>, <32 x i32>)
define void @V6_vscattermwq_128B(<32 x i32> %a, i32 %b, i32 %c, <32 x i32> %d, <32 x i32> %e) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vscattermwq.128B(<128 x i1> %1, i32 %b, i32 %c, <32 x i32> %d, <32 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhq.128B(<128 x i1>, i32, i32, <32 x i32>, <32 x i32>)
define void @V6_vscattermhq_128B(<32 x i32> %a, i32 %b, i32 %c, <32 x i32> %d, <32 x i32> %e) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vscattermhq.128B(<128 x i1> %1, i32 %b, i32 %c, <32 x i32> %d, <32 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhw.128B(i32, i32, <64 x i32>, <32 x i32>)
define void @V6_vscattermhw_128B(i32 %a, i32 %b, <64 x i32> %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermhw.128B(i32 %a, i32 %b, <64 x i32> %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhw.add.128B(i32, i32, <64 x i32>, <32 x i32>)
define void @V6_vscattermhw_add_128B(i32 %a, i32 %b, <64 x i32> %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermhw.add.128B(i32 %a, i32 %b, <64 x i32> %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhwq.128B(<128 x i1>, i32, i32, <64 x i32>, <32 x i32>)
define void @V6_vscattermhwq_128B(<32 x i32> %a, i32 %b, i32 %c, <64 x i32> %d, <32 x i32> %e) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vscattermhwq.128B(<128 x i1> %1, i32 %b, i32 %c, <64 x i32> %d, <32 x i32> %e)
  ret void
}
