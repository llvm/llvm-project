; RUN: llc -mattr=+hvxv60,hvx-length64b -march=hexagon -O2 < %s | FileCheck %s

; CHECK-LABEL: V6_vmaskedstoreq
; CHECK: if (q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorenq
; CHECK: if (!q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorentq
; CHECK: if (q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorentnq
; CHECK: if (!q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

declare <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32>, i32)

declare void @llvm.hexagon.V6.vmaskedstoreq(<64 x i1>, i8*, <16 x i32>)
define void @V6_vmaskedstoreq( <16 x i32> %a, i8* %b, <16 x i32> %c) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstoreq(<64 x i1> %1, i8* %b, <16 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorenq(<64 x i1>, i8*, <16 x i32>)
define void @V6_vmaskedstorenq( <16 x i32> %a, i8* %b, <16 x i32> %c) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstorenq(<64 x i1> %1, i8* %b, <16 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorentq(<64 x i1>, i8*, <16 x i32>)
define void @V6_vmaskedstorentq( <16 x i32> %a, i8* %b, <16 x i32> %c) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstorentq(<64 x i1> %1, i8* %b, <16 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorentnq(<64 x i1>, i8*, <16 x i32>)
define void @V6_vmaskedstorentnq( <16 x i32> %a, i8* %b, <16 x i32> %c) {
  %1 = tail call <64 x i1> @llvm.hexagon.V6.vandvrt(<16 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstorentnq(<64 x i1> %1, i8* %b, <16 x i32> %c)
  ret void
}
