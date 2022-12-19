; RUN: llc -mv65 -mattr=+hvxv65,hvx-length64b -march=hexagon -O2 < %s | FileCheck %s
; RUN: llc -mv65 -mattr=+hvxv65,hvx-length64b -march=hexagon -O2 -disable-packetizer < %s | FileCheck %s
; RUN: llc -mv65 -mattr=+hvxv65,hvx-length64b -march=hexagon -O0 < %s | FileCheck %s

; CHECK: vtmp.h = vgather(r{{[0-9]+}},m{{[0-9]+}},v{{[0-9]+}}.h).h
; CHECK-NEXT: vmem(r{{[0-9]+}}+#0) = vtmp.new
; CHECK-NEXT: }

declare i32 @add_translation_extended(i32, ptr, i64, i32, i32, i32, i32, i32, i32) local_unnamed_addr

; Function Attrs: nounwind
define i32 @main() local_unnamed_addr {
entry:
  %hvx_vector = alloca <16 x i32>, align 64
  %call.i = tail call i32 @add_translation_extended(i32 1, ptr inttoptr (i32 -668991488 to ptr), i64 3625975808, i32 16, i32 15, i32 0, i32 0, i32 0, i32 3)
  %0 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 1)
  %1 = tail call <16 x i32> @llvm.hexagon.V6.lvsplatw(i32 2)
  tail call void @llvm.hexagon.V6.vscattermh.add(i32 -668991488, i32 1023, <16 x i32> %0, <16 x i32> %1)
  call void @llvm.hexagon.V6.vgathermh(ptr %hvx_vector, i32 -668991488, i32 1023, <16 x i32> %0)
  ret i32 0
}

; Function Attrs: nounwind writeonly
declare void @llvm.hexagon.V6.vscattermh.add(i32, i32, <16 x i32>, <16 x i32>)

; Function Attrs: nounwind readnone
declare <16 x i32> @llvm.hexagon.V6.lvsplatw(i32)

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vgathermh(ptr, i32, i32, <16 x i32>)

