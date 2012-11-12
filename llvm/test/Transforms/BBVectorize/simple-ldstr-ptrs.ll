target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -instcombine -gvn -S | FileCheck %s
; RUN: opt < %s -bb-vectorize -bb-vectorize-req-chain-depth=3 -bb-vectorize-aligned-only -instcombine -gvn -S | FileCheck %s -check-prefix=CHECK-AO

; FIXME: re-enable this once pointer vectors work properly
; XFAIL: *

; Simple 3-pair chain also with loads and stores (using ptrs and gep)
define double @test1(i64* %a, i64* %b, i64* %c) nounwind uwtable readonly {
entry:
  %i0 = load i64* %a, align 8
  %i1 = load i64* %b, align 8
  %mul = mul i64 %i0, %i1
  %arrayidx3 = getelementptr inbounds i64* %a, i64 1
  %i3 = load i64* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds i64* %b, i64 1
  %i4 = load i64* %arrayidx4, align 8
  %mul5 = mul i64 %i3, %i4
  %ptr = inttoptr i64 %mul to double*
  %ptr5 = inttoptr i64 %mul5 to double*
  %aptr = getelementptr inbounds double* %ptr, i64 2
  %aptr5 = getelementptr inbounds double* %ptr5, i64 3
  %av = load double* %aptr, align 16
  %av5 = load double* %aptr5, align 16
  %r = fmul double %av, %av5
  store i64 %mul, i64* %c, align 8
  %arrayidx5 = getelementptr inbounds i64* %c, i64 1
  store i64 %mul5, i64* %arrayidx5, align 8
  ret double %r
; CHECK: @test1
; CHECK: %i0.v.i0 = bitcast i64* %a to <2 x i64>*
; CHECK: %i1.v.i0 = bitcast i64* %b to <2 x i64>*
; CHECK: %i0 = load <2 x i64>* %i0.v.i0, align 8
; CHECK: %i1 = load <2 x i64>* %i1.v.i0, align 8
; CHECK: %mul = mul <2 x i64> %i0, %i1
; CHECK: %ptr = inttoptr <2 x i64> %mul to <2 x double*>
; CHECK: %aptr = getelementptr inbounds <2 x double*> %ptr, <2 x i64> <i64 2, i64 3>
; CHECK: %aptr.v.r1 = extractelement <2 x double*> %aptr, i32 0
; CHECK: %aptr.v.r2 = extractelement <2 x double*> %aptr, i32 1
; CHECK: %av = load double* %aptr.v.r1, align 16
; CHECK: %av5 = load double* %aptr.v.r2, align 16
; CHECK: %r = fmul double %av, %av5
; CHECK: %0 = bitcast i64* %c to <2 x i64>*
; CHECK: store <2 x i64> %mul, <2 x i64>* %0, align 8
; CHECK: ret double %r
; CHECK-AO: @test1
; CHECK-AO-NOT: load <2 x
}

; Simple 3-pair chain with loads and stores (using ptrs and gep)
define void @test2(i64** %a, i64** %b, i64** %c) nounwind uwtable readonly {
entry:
  %i0 = load i64** %a, align 8
  %i1 = load i64** %b, align 8
  %arrayidx3 = getelementptr inbounds i64** %a, i64 1
  %i3 = load i64** %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds i64** %b, i64 1
  %i4 = load i64** %arrayidx4, align 8
  %o1 = load i64* %i1, align 8
  %o4 = load i64* %i4, align 8
  %ptr0 = getelementptr inbounds i64* %i0, i64 %o1
  %ptr3 = getelementptr inbounds i64* %i3, i64 %o4
  store i64* %ptr0, i64** %c, align 8
  %arrayidx5 = getelementptr inbounds i64** %c, i64 1
  store i64* %ptr3, i64** %arrayidx5, align 8
  ret void
; CHECK: @test2
; CHECK: %i0.v.i0 = bitcast i64** %a to <2 x i64*>*
; CHECK: %i1 = load i64** %b, align 8
; CHECK: %i0 = load <2 x i64*>* %i0.v.i0, align 8
; CHECK: %arrayidx4 = getelementptr inbounds i64** %b, i64 1
; CHECK: %i4 = load i64** %arrayidx4, align 8
; CHECK: %o1 = load i64* %i1, align 8
; CHECK: %o4 = load i64* %i4, align 8
; CHECK: %ptr0.v.i1.1 = insertelement <2 x i64> undef, i64 %o1, i32 0
; CHECK: %ptr0.v.i1.2 = insertelement <2 x i64> %ptr0.v.i1.1, i64 %o4, i32 1
; CHECK: %ptr0 = getelementptr inbounds <2 x i64*> %i0, <2 x i64> %ptr0.v.i1.2
; CHECK: %0 = bitcast i64** %c to <2 x i64*>*
; CHECK: store <2 x i64*> %ptr0, <2 x i64*>* %0, align 8
; CHECK: ret void
; CHECK-AO: @test2
; CHECK-AO-NOT: <2 x
}

; Simple 3-pair chain with loads and stores (using ptrs and gep)
; using pointer vectors.
define void @test3(<2 x i64*>* %a, <2 x i64*>* %b, <2 x i64*>* %c) nounwind uwtable readonly {
entry:
  %i0 = load <2 x i64*>* %a, align 8
  %i1 = load <2 x i64*>* %b, align 8
  %arrayidx3 = getelementptr inbounds <2 x i64*>* %a, i64 1
  %i3 = load <2 x i64*>* %arrayidx3, align 8
  %arrayidx4 = getelementptr inbounds <2 x i64*>* %b, i64 1
  %i4 = load <2 x i64*>* %arrayidx4, align 8
  %j1 = extractelement <2 x i64*> %i1, i32 0
  %j4 = extractelement <2 x i64*> %i4, i32 0
  %o1 = load i64* %j1, align 8
  %o4 = load i64* %j4, align 8
  %j0 = extractelement <2 x i64*> %i0, i32 0
  %j3 = extractelement <2 x i64*> %i3, i32 0
  %ptr0 = getelementptr inbounds i64* %j0, i64 %o1
  %ptr3 = getelementptr inbounds i64* %j3, i64 %o4
  %qtr0 = insertelement <2 x i64*> undef, i64* %ptr0, i32 0
  %rtr0 = insertelement <2 x i64*> %qtr0, i64* %ptr0, i32 1
  %qtr3 = insertelement <2 x i64*> undef, i64* %ptr3, i32 0
  %rtr3 = insertelement <2 x i64*> %qtr3, i64* %ptr3, i32 1
  store <2 x i64*> %rtr0, <2 x i64*>* %c, align 8
  %arrayidx5 = getelementptr inbounds <2 x i64*>* %c, i64 1
  store <2 x i64*> %rtr3, <2 x i64*>* %arrayidx5, align 8
  ret void
; CHECK: @test3
; CHECK: %i0.v.i0 = bitcast <2 x i64*>* %a to <4 x i64*>*
; CHECK: %i1 = load <2 x i64*>* %b, align 8
; CHECK: %i0 = load <4 x i64*>* %i0.v.i0, align 8
; CHECK: %arrayidx4 = getelementptr inbounds <2 x i64*>* %b, i64 1
; CHECK: %i4 = load <2 x i64*>* %arrayidx4, align 8
; CHECK: %j1 = extractelement <2 x i64*> %i1, i32 0
; CHECK: %j4 = extractelement <2 x i64*> %i4, i32 0
; CHECK: %o1 = load i64* %j1, align 8
; CHECK: %o4 = load i64* %j4, align 8
; CHECK: %ptr0.v.i1.1 = insertelement <2 x i64> undef, i64 %o1, i32 0
; CHECK: %ptr0.v.i1.2 = insertelement <2 x i64> %ptr0.v.i1.1, i64 %o4, i32 1
; CHECK: %ptr0.v.i0 = shufflevector <4 x i64*> %i0, <4 x i64*> undef, <2 x i32> <i32 0, i32 2>
; CHECK: %ptr0 = getelementptr inbounds <2 x i64*> %ptr0.v.i0, <2 x i64> %ptr0.v.i1.2
; CHECK: %rtr0 = shufflevector <2 x i64*> %ptr0, <2 x i64*> undef, <2 x i32> zeroinitializer
; CHECK: %rtr3 = shufflevector <2 x i64*> %ptr0, <2 x i64*> undef, <2 x i32> <i32 1, i32 1>
; CHECK: %0 = bitcast <2 x i64*>* %c to <4 x i64*>*
; CHECK: %1 = shufflevector <2 x i64*> %rtr0, <2 x i64*> %rtr3, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK: store <4 x i64*> %1, <4 x i64*>* %0, align 8
; CHECK: ret void
; CHECK-AO: @test3
; CHECK-AO-NOT: <4 x
}

