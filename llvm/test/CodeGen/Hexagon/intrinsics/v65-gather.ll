; RUN: llc -mv65 -mattr=+hvxv65,hvx-length64b -march=hexagon -O2 < %s | FileCheck %s

; CHECK-LABEL: V6_vgathermw
; CHECK: vtmp.w = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.w).w
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermh
; CHECK: vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.h).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermhw
; CHECK: vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermwq
; CHECK: if (q{{[0-3]+}}) vtmp.w = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.w).w
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermhq
; CHECK: if (q{{[0-3]+}}) vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}.h).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-LABEL: V6_vgathermhwq
; CHECK: if (q{{[0-3]+}}) vtmp.h = vgather(r1,m{{[0-9]+}},v{{[0-9]+}}:{{[0-9]+}}.w).h
; CHECK: vmem(r{{[0-9]+}}+#0) = vtmp.new

declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32)

declare void @llvm.hexagon.V6.vgathermw(i8*, i32, i32, <16 x i32>)
define void @V6_vgathermw(i8* %a, i32 %b, i32 %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vgathermw(i8* %a, i32 %b, i32 %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vgathermh(i8*, i32, i32, <16 x i32>)
define void @V6_vgathermh(i8* %a, i32 %b, i32 %c, <16 x i32> %d) {
  call void @llvm.hexagon.V6.vgathermh(i8* %a, i32 %b, i32 %c, <16 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vgathermhw(i8*, i32, i32, <32 x i32>)
define void @V6_vgathermhw(i8* %a, i32 %b, i32 %c, <32 x i32> %d) {
  call void @llvm.hexagon.V6.vgathermhw(i8* %a, i32 %b, i32 %c, <32 x i32> %d)
  ret void
}

declare void @llvm.hexagon.V6.vgathermwq(i8*, <64 x i1>, i32, i32, <16 x i32>)
define void @V6_vgathermwq(i8* %a, <16 x i32> %b, i32 %c, i32 %d, <16 x i32> %e) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  call void @llvm.hexagon.V6.vgathermwq(i8* %a, <64 x i1> %1, i32 %c, i32 %d, <16 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vgathermhq(i8*, <64 x i1>, i32, i32, <16 x i32>)
define void @V6_vgathermhq(i8* %a, <16 x i32> %b, i32 %c, i32 %d, <16 x i32> %e) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  call void @llvm.hexagon.V6.vgathermhq(i8* %a, <64 x i1> %1, i32 %c, i32 %d, <16 x i32> %e)
  ret void
}

declare void @llvm.hexagon.V6.vgathermhwq(i8*, <64 x i1>, i32, i32, <32 x i32>)
define void @V6_vgathermhwq(i8* %a, <16 x i32> %b, i32 %c, i32 %d, <32 x i32> %e) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %b, i32 -1)
  call void @llvm.hexagon.V6.vgathermhwq(i8* %a, <64 x i1> %1, i32 %c, i32 %d, <32 x i32> %e)
  ret void
}
