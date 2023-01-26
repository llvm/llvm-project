; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: vmemu
; CHECK: vmux
define <128 x i8> @f0(ptr %a0, i32 %a1, i32 %a2) #0 {
  %q0 = call <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32 %a2)
  %v0 = call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 %a1)
  %v1 = bitcast <32 x i32> %v0 to <128 x i8>
  %v2 = call <128 x i8> @llvm.masked.load.v128i8.p0(ptr %a0, i32 4, <128 x i1> %q0, <128 x i8> %v1)
  ret <128 x i8> %v2
}

; CHECK-LABEL: f1:
; CHECK: vlalign
; CHECK: if (q{{.}}) vmem{{.*}} = v
define void @f1(ptr %a0, i32 %a1, i32 %a2) #0 {
  %q0 = call <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32 %a2)
  %v0 = call <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32 %a1)
  %v1 = bitcast <32 x i32> %v0 to <128 x i8>
  call void @llvm.masked.store.v128i8.p0(<128 x i8> %v1, ptr %a0, i32 4, <128 x i1> %q0)
  ret void
}

declare <128 x i1> @llvm.hexagon.V6.pred.scalar2.128B(i32) #1
declare <32 x i32> @llvm.hexagon.V6.lvsplatb.128B(i32) #1
declare <128 x i8> @llvm.masked.load.v128i8.p0(ptr, i32 immarg, <128 x i1>, <128 x i8>) #2
declare void @llvm.masked.store.v128i8.p0(<128 x i8>, ptr, i32 immarg, <128 x i1>) #2

attributes #0 = { nounwind "target-features"="+hvxv65,+hvx-length128b" }
attributes #1 = { nounwind readnone }
attributes #2 = { argmemonly nounwind readonly willreturn }
attributes #3 = { argmemonly nounwind willreturn }


