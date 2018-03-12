; RUN: llc -march=hexagon -mattr=+hvxv60,+hvx-length128b < %s | FileCheck %s

; Test that we generate code for the vector byte enable store instrinsics.

; CHECK-LABEL: f0:
; CHECK: if (q{{[0-3]}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

define void @f0(<32 x i32> %a0, i8* %a1, <32 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <32 x i32> %a0 to <1024 x i1>
  tail call void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<1024 x i1> %v0, i8* %a1, <32 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.qpred.ai.128B(<1024 x i1>, i8*, <32 x i32>) #0

; CHECK-LABEL: f1:
; CHECK: if (!q{{[0-3]}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

define void @f1(<32 x i32> %a0, i8* %a1, <32 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <32 x i32> %a0 to <1024 x i1>
  tail call void @llvm.hexagon.V6.vS32b.nqpred.ai.128B(<1024 x i1> %v0, i8* %a1, <32 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.nqpred.ai.128B(<1024 x i1>, i8*, <32 x i32>) #0

; CHECK-LABEL: f2:
; CHECK: if (q{{[0-3]}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

define void @f2(<32 x i32> %a0, i8* %a1, <32 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <32 x i32> %a0 to <1024 x i1>
  tail call void @llvm.hexagon.V6.vS32b.nt.qpred.ai.128B(<1024 x i1> %v0, i8* %a1, <32 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.nt.qpred.ai.128B(<1024 x i1>, i8*, <32 x i32>) #0

; CHECK-LABEL: f3:
; CHECK: if (!q{{[0-3]}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

define void @f3(<32 x i32> %a0, i8* %a1, <32 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <32 x i32> %a0 to <1024 x i1>
  tail call void @llvm.hexagon.V6.vS32b.nt.nqpred.ai.128B(<1024 x i1> %v0, i8* %a1, <32 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.nt.nqpred.ai.128B(<1024 x i1>, i8*, <32 x i32>) #0

attributes #0 = { argmemonly nounwind }
