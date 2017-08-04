; RUN: llc -mattr=+hvx-double -march=hexagon -O2 < %s | FileCheck %s

; CHECK-LABEL: V6_vmaskedstoreq_128B
; CHECK: if (q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorenq_128B
; CHECK: if (!q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorentq_128B
; CHECK: if (q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

; CHECK-LABEL: V6_vmaskedstorentnq_128B
; CHECK: if (!q{{[0-3]+}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

declare void @llvm.hexagon.V6.vmaskedstoreq.128B(<1024 x i1>, i8*, <32 x i32>)
define void @V6_vmaskedstoreq_128B( <32 x i32> %a, i8* %b, <32 x i32> %c) {
  %1 = bitcast <32 x i32> %a to <1024 x i1>
  call void @llvm.hexagon.V6.vmaskedstoreq.128B(<1024 x i1> %1, i8* %b, <32 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorenq.128B(<1024 x i1>, i8*, <32 x i32>)
define void @V6_vmaskedstorenq_128B( <32 x i32> %a, i8* %b, <32 x i32> %c) {
  %1 = bitcast <32 x i32> %a to <1024 x i1>
  call void @llvm.hexagon.V6.vmaskedstorenq.128B(<1024 x i1> %1, i8* %b, <32 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorentq.128B(<1024 x i1>, i8*, <32 x i32>)
define void @V6_vmaskedstorentq_128B( <32 x i32> %a, i8* %b, <32 x i32> %c) {
  %1 = bitcast <32 x i32> %a to <1024 x i1>
  call void @llvm.hexagon.V6.vmaskedstorentq.128B(<1024 x i1> %1, i8* %b, <32 x i32> %c)
  ret void
}

declare void @llvm.hexagon.V6.vmaskedstorentnq.128B(<1024 x i1>, i8*, <32 x i32>)
define void @V6_vmaskedstorentnq_128B( <32 x i32> %a, i8* %b, <32 x i32> %c) {
  %1 = bitcast <32 x i32> %a to <1024 x i1>
  call void @llvm.hexagon.V6.vmaskedstorentnq.128B(<1024 x i1> %1, i8* %b, <32 x i32> %c)
  ret void
}
