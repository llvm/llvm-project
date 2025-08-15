; RUN: llc -mv65 -mattr=+hvxv65,hvx-length64b -mtriple=hexagon -O2 < %s | FileCheck %s

; CHECK-LABEL: V6_vscattermw
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.w).w = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermh
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.h).h = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermw_add
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.w).w += v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermh_add
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.h).h += v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermwq
; CHECK: if (q{{[0-3]}}) vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.w).w = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhq
; CHECK: if (q{{[0-3]}}) vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.h).h = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhw
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h = v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhw_add
; CHECK: vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h += v{{[0-9]+}}
; CHECK-LABEL: V6_vscattermhwq
; CHECK: if (q{{[0-3]}}) vscatter(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h = v{{[0-9]+}}

declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32)

declare void @llvm.hexagon.V6.vscattermw(i32, i32, <16 x i32>, <16 x i32>)
define void @V6_vscattermw(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermw(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermh(i32, i32, <16 x i32>, <16 x i32>)
define void @V6_vscattermh(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermh(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermw.add(i32, i32, <16 x i32>, <16 x i32>)
define void @V6_vscattermw_add(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermw.add(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermh.add(i32, i32, <16 x i32>, <16 x i32>)
define void @V6_vscattermh_add(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermh.add(i32 %a, i32 %b, <16 x i32> %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermwq(<64 x i1>, i32, i32, <16 x i32>, <16 x i32>)
define void @V6_vscattermwq(<16 x i32> %a, i32 %b, i32 %c, <16 x i32> %d, <16 x i32> %e) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vscattermwq(<64 x i1> %1, i32 %b, i32 %c, <16 x i32> %d, <16 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhq(<64 x i1>, i32, i32, <16 x i32>, <16 x i32>)
define void @V6_vscattermhq(<16 x i32> %a, i32 %b, i32 %c, <16 x i32> %d, <16 x i32> %e) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vscattermhq(<64 x i1> %1, i32 %b, i32 %c, <16 x i32> %d, <16 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhw(i32, i32, <32 x i32>, <16 x i32>)
define void @V6_vscattermhw(i32 %a, i32 %b, <32 x i32> %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermhw(i32 %a, i32 %b, <32 x i32> %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhw.add(i32, i32, <32 x i32>, <16 x i32>)
define void @V6_vscattermhw_add(i32 %a, i32 %b, <32 x i32> %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vscattermhw.add(i32 %a, i32 %b, <32 x i32> %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vscattermhwq(<64 x i1>, i32, i32, <32 x i32>, <16 x i32>)
define void @V6_vscattermhwq(<16 x i32> %a, i32 %b, i32 %c, <32 x i32> %d, <16 x i32> %e) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vscattermhwq(<64 x i1> %1, i32 %b, i32 %c, <32 x i32> %d, <16 x i32> %e)
  ret void
}
