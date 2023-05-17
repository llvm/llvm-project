; RUN: llc -mattr=+hvxv60,hvx-length128b -march=hexagon -O2 < %s | FileCheck %s

; CHECK-LABEL: V6_vmaskedstoreq_128B
; CHECK: if (q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorenq_128B
; CHECK: if (!q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorentq_128B
; CHECK: if (q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorentnq_128B
; CHECK: if (!q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

declare <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32>, i32)

declare void @llvm.hexagon.V6.vmaskedstoreq.128B(<128 x i1>, ptr, <32 x i32>)
define void @V6_vmaskedstoreq_128B( <32 x i32> %a, ptr %b, <32 x i32> %c) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstoreq.128B(<128 x i1> %1, ptr %b, <32 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorenq.128B(<128 x i1>, ptr, <32 x i32>)
define void @V6_vmaskedstorenq_128B( <32 x i32> %a, ptr %b, <32 x i32> %c) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstorenq.128B(<128 x i1> %1, ptr %b, <32 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorentq.128B(<128 x i1>, ptr, <32 x i32>)
define void @V6_vmaskedstorentq_128B( <32 x i32> %a, ptr %b, <32 x i32> %c) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstorentq.128B(<128 x i1> %1, ptr %b, <32 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorentnq.128B(<128 x i1>, ptr, <32 x i32>)
define void @V6_vmaskedstorentnq_128B( <32 x i32> %a, ptr %b, <32 x i32> %c) {
  %1 = tail call <128 x i1> @llvm.hexagon.V6.vandvrt.128B(<32 x i32> %a, i32 -1)
  call void @llvm.hexagon.V6.vmaskedstorentnq.128B(<128 x i1> %1, ptr %b, <32 x i32> %c)
  ret void
}
