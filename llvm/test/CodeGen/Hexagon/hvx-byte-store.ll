; RUN: llc -march=hexagon -mattr=+hvxv60,+hvx-length64b < %s | FileCheck %s

; Test that we generate code for the vector byte enabled store intrinsics.

; CHECK-LABEL: f0:
; CHECK: if (q{{[0-3]}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

define void @f0(<16 x i32> %a0, i8* %a1, <16 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <16 x i32> %a0 to <512 x i1>
  tail call void @llvm.hexagon.V6.vS32b.qpred.ai(<512 x i1> %v0, i8* %a1, <16 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.qpred.ai(<512 x i1>, i8*, <16 x i32>) #0

; CHECK-LABEL: f1:
; CHECK: if (!q{{[0-3]}}) vmem(r{{[0-9]+}}+#0) = v{{[0-9]+}}

define void @f1(<16 x i32> %a0, i8* %a1, <16 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <16 x i32> %a0 to <512 x i1>
  tail call void @llvm.hexagon.V6.vS32b.nqpred.ai(<512 x i1> %v0, i8* %a1, <16 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.nqpred.ai(<512 x i1>, i8*, <16 x i32>) #0

; CHECK-LABEL: f2:
; CHECK: if (q{{[0-3]}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

define void @f2(<16 x i32> %a0, i8* %a1, <16 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <16 x i32> %a0 to <512 x i1>
  tail call void @llvm.hexagon.V6.vS32b.nt.qpred.ai(<512 x i1> %v0, i8* %a1, <16 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.nt.qpred.ai(<512 x i1>, i8*, <16 x i32>) #0

; CHECK-LABEL: f3:
; CHECK: if (!q{{[0-3]}}) vmem(r{{[0-9]+}}+#0):nt = v{{[0-9]+}}

define void @f3(<16 x i32> %a0, i8* %a1, <16 x i32> %a2) local_unnamed_addr {
b0:
  %v0 = bitcast <16 x i32> %a0 to <512 x i1>
  tail call void @llvm.hexagon.V6.vS32b.nt.nqpred.ai(<512 x i1> %v0, i8* %a1, <16 x i32> %a2)
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.hexagon.V6.vS32b.nt.nqpred.ai(<512 x i1>, i8*, <16 x i32>) #0

attributes #0 = { argmemonly nounwind }
