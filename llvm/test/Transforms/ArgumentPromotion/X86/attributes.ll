; RUN: opt -S -argpromotion < %s | FileCheck %s
; RUN: opt -S -passes=argpromotion < %s | FileCheck %s
; Test that we only promote arguments when the caller/callee have compatible
; function attrubtes.

target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @no_promote_avx2(<4 x i64>* %arg, <4 x i64>* readonly %arg1)
define internal fastcc void @no_promote_avx2(<4 x i64>* %arg, <4 x i64>* readonly %arg1) #0 {
bb:
  %tmp = load <4 x i64>, <4 x i64>* %arg1
  store <4 x i64> %tmp, <4 x i64>* %arg
  ret void
}

define void @no_promote(<4 x i64>* %arg) #1 {
bb:
  %tmp = alloca <4 x i64>, align 32
  %tmp2 = alloca <4 x i64>, align 32
  %tmp3 = bitcast <4 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @no_promote_avx2(<4 x i64>* %tmp2, <4 x i64>* %tmp)
  %tmp4 = load <4 x i64>, <4 x i64>* %tmp2, align 32
  store <4 x i64> %tmp4, <4 x i64>* %arg, align 2
  ret void
}

; CHECK-LABEL: @promote_avx2(<4 x i64>* %arg, <4 x i64> %
define internal fastcc void @promote_avx2(<4 x i64>* %arg, <4 x i64>* readonly %arg1) #0 {
bb:
  %tmp = load <4 x i64>, <4 x i64>* %arg1
  store <4 x i64> %tmp, <4 x i64>* %arg
  ret void
}

define void @promote(<4 x i64>* %arg) #0 {
bb:
  %tmp = alloca <4 x i64>, align 32
  %tmp2 = alloca <4 x i64>, align 32
  %tmp3 = bitcast <4 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @promote_avx2(<4 x i64>* %tmp2, <4 x i64>* %tmp)
  %tmp4 = load <4 x i64>, <4 x i64>* %tmp2, align 32
  store <4 x i64> %tmp4, <4 x i64>* %arg, align 2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #2

attributes #0 = { inlinehint norecurse nounwind uwtable "target-features"="+avx2" }
attributes #1 = { nounwind uwtable }
attributes #2 = { argmemonly nounwind }
