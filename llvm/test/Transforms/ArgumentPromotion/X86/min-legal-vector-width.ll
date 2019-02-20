; RUN: opt -S -argpromotion < %s | FileCheck %s
; RUN: opt -S -passes=argpromotion < %s | FileCheck %s
; Test that we only promote arguments when the caller/callee have compatible
; function attrubtes.

target triple = "x86_64-unknown-linux-gnu"

; This should promote
; CHECK-LABEL: @callee_avx512_legal512_prefer512_call_avx512_legal512_prefer512(<8 x i64>* %arg, <8 x i64> %arg1.val)
define internal fastcc void @callee_avx512_legal512_prefer512_call_avx512_legal512_prefer512(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #0 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx512_legal512_prefer512_call_avx512_legal512_prefer512(<8 x i64>* %arg) #0 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx512_legal512_prefer512_call_avx512_legal512_prefer512(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; This should promote
; CHECK-LABEL: @callee_avx512_legal512_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %arg, <8 x i64> %arg1.val)
define internal fastcc void @callee_avx512_legal512_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #1 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx512_legal512_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %arg) #1 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx512_legal512_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; This should promote
; CHECK-LABEL: @callee_avx512_legal512_prefer512_call_avx512_legal512_prefer256(<8 x i64>* %arg, <8 x i64> %arg1.val)
define internal fastcc void @callee_avx512_legal512_prefer512_call_avx512_legal512_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #1 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx512_legal512_prefer512_call_avx512_legal512_prefer256(<8 x i64>* %arg) #0 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx512_legal512_prefer512_call_avx512_legal512_prefer256(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; This should promote
; CHECK-LABEL: @callee_avx512_legal512_prefer256_call_avx512_legal512_prefer512(<8 x i64>* %arg, <8 x i64> %arg1.val)
define internal fastcc void @callee_avx512_legal512_prefer256_call_avx512_legal512_prefer512(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #0 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx512_legal512_prefer256_call_avx512_legal512_prefer512(<8 x i64>* %arg) #1 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx512_legal512_prefer256_call_avx512_legal512_prefer512(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; This should not promote
; CHECK-LABEL: @callee_avx512_legal256_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1)
define internal fastcc void @callee_avx512_legal256_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #1 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx512_legal256_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %arg) #2 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx512_legal256_prefer256_call_avx512_legal512_prefer256(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; This should not promote
; CHECK-LABEL: @callee_avx512_legal512_prefer256_call_avx512_legal256_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1)
define internal fastcc void @callee_avx512_legal512_prefer256_call_avx512_legal256_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #2 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx512_legal512_prefer256_call_avx512_legal256_prefer256(<8 x i64>* %arg) #1 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx512_legal512_prefer256_call_avx512_legal256_prefer256(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; This should promote
; CHECK-LABEL: @callee_avx2_legal256_prefer256_call_avx2_legal512_prefer256(<8 x i64>* %arg, <8 x i64> %arg1.val)
define internal fastcc void @callee_avx2_legal256_prefer256_call_avx2_legal512_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #3 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx2_legal256_prefer256_call_avx2_legal512_prefer256(<8 x i64>* %arg) #4 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx2_legal256_prefer256_call_avx2_legal512_prefer256(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; This should promote
; CHECK-LABEL: @callee_avx2_legal512_prefer256_call_avx2_legal256_prefer256(<8 x i64>* %arg, <8 x i64> %arg1.val)
define internal fastcc void @callee_avx2_legal512_prefer256_call_avx2_legal256_prefer256(<8 x i64>* %arg, <8 x i64>* readonly %arg1) #4 {
bb:
  %tmp = load <8 x i64>, <8 x i64>* %arg1
  store <8 x i64> %tmp, <8 x i64>* %arg
  ret void
}

define void @avx2_legal512_prefer256_call_avx2_legal256_prefer256(<8 x i64>* %arg) #3 {
bb:
  %tmp = alloca <8 x i64>, align 32
  %tmp2 = alloca <8 x i64>, align 32
  %tmp3 = bitcast <8 x i64>* %tmp to i8*
  call void @llvm.memset.p0i8.i64(i8* align 32 %tmp3, i8 0, i64 32, i1 false)
  call fastcc void @callee_avx2_legal512_prefer256_call_avx2_legal256_prefer256(<8 x i64>* %tmp2, <8 x i64>* %tmp)
  %tmp4 = load <8 x i64>, <8 x i64>* %tmp2, align 32
  store <8 x i64> %tmp4, <8 x i64>* %arg, align 2
  ret void
}

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i1) #5

attributes #0 = { inlinehint norecurse nounwind uwtable "target-features"="+avx512vl" "min-legal-vector-width"="512" "prefer-vector-width"="512" }
attributes #1 = { inlinehint norecurse nounwind uwtable "target-features"="+avx512vl" "min-legal-vector-width"="512" "prefer-vector-width"="256" }
attributes #2 = { inlinehint norecurse nounwind uwtable "target-features"="+avx512vl" "min-legal-vector-width"="256" "prefer-vector-width"="256" }
attributes #3 = { inlinehint norecurse nounwind uwtable "target-features"="+avx2" "min-legal-vector-width"="512" "prefer-vector-width"="256" }
attributes #4 = { inlinehint norecurse nounwind uwtable "target-features"="+avx2" "min-legal-vector-width"="256" "prefer-vector-width"="256" }
attributes #5 = { argmemonly nounwind }
