; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

%S1 = type { i64, [42 x float] }

define i32 @test1(<4 x i32> %x, <4 x i32> %y) {
; CHECK-LABEL: @test1(
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %tmp1 = load i32, i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32, i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32, i32* %a.tmp3
; CHECK-NOT: load
; CHECK:      extractelement <4 x i32> %x, i32 2
; CHECK-NEXT: extractelement <4 x i32> %y, i32 3
; CHECK-NEXT: extractelement <4 x i32> %y, i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

define i32 @test2(<4 x i32> %x, <4 x i32> %y) {
; CHECK-LABEL: @test2(
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %tmp1 = load i32, i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32, i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %a.tmp3.cast = bitcast i32* %a.tmp3 to <2 x i32>*
  %tmp3.vec = load <2 x i32>, <2 x i32>* %a.tmp3.cast
  %tmp3 = extractelement <2 x i32> %tmp3.vec, i32 0
; CHECK-NOT: load
; CHECK:      %[[extract1:.*]] = extractelement <4 x i32> %x, i32 2
; CHECK-NEXT: %[[extract2:.*]] = extractelement <4 x i32> %y, i32 3
; CHECK-NEXT: %[[extract3:.*]] = shufflevector <4 x i32> %y, <4 x i32> undef, <2 x i32> <i32 0, i32 1>
; CHECK-NEXT: %[[extract4:.*]] = extractelement <2 x i32> %[[extract3]], i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
; CHECK-NEXT: %[[sum1:.*]] = add i32 %[[extract1]], %[[extract2]]
; CHECK-NEXT: %[[sum2:.*]] = add i32 %[[extract4]], %[[sum1]]
; CHECK-NEXT: ret i32 %[[sum2]]
}

define i32 @test3(<4 x i32> %x, <4 x i32> %y) {
; CHECK-LABEL: @test3(
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.y.cast = bitcast <4 x i32>* %a.y to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.y.cast, i8 0, i32 16, i32 1, i1 false)
; CHECK-NOT: memset

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %a.tmp1.cast = bitcast i32* %a.tmp1 to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.tmp1.cast, i8 -1, i32 4, i32 1, i1 false)
  %tmp1 = load i32, i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32, i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32, i32* %a.tmp3
; CHECK-NOT: load
; CHECK:      %[[insert:.*]] = insertelement <4 x i32> %x, i32 -1, i32 2
; CHECK-NEXT: extractelement <4 x i32> %[[insert]], i32 2
; CHECK-NEXT: extractelement <4 x i32> zeroinitializer, i32 3
; CHECK-NEXT: extractelement <4 x i32> zeroinitializer, i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

define i32 @test4(<4 x i32> %x, <4 x i32> %y, <4 x i32>* %z) {
; CHECK-LABEL: @test4(
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.y.cast = bitcast <4 x i32>* %a.y to i8*
  %z.cast = bitcast <4 x i32>* %z to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.y.cast, i8* %z.cast, i32 16, i32 1, i1 false)
; CHECK-NOT: memcpy

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %a.tmp1.cast = bitcast i32* %a.tmp1 to i8*
  %z.tmp1 = getelementptr inbounds <4 x i32>, <4 x i32>* %z, i64 0, i64 2
  %z.tmp1.cast = bitcast i32* %z.tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.tmp1.cast, i8* %z.tmp1.cast, i32 4, i32 1, i1 false)
  %tmp1 = load i32, i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32, i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32, i32* %a.tmp3
; CHECK-NOT: memcpy
; CHECK:      %[[load:.*]] = load <4 x i32>, <4 x i32>* %z
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* %z, i64 0, i64 2
; CHECK-NEXT: %[[element_load:.*]] = load i32, i32* %[[gep]]
; CHECK-NEXT: %[[insert:.*]] = insertelement <4 x i32> %x, i32 %[[element_load]], i32 2
; CHECK-NEXT: extractelement <4 x i32> %[[insert]], i32 2
; CHECK-NEXT: extractelement <4 x i32> %[[load]], i32 3
; CHECK-NEXT: extractelement <4 x i32> %[[load]], i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

declare void @llvm.memcpy.p0i8.p1i8.i32(i8* nocapture, i8 addrspace(1)* nocapture, i32, i32, i1) nounwind

; Same as test4 with a different sized address  space pointer source.
define i32 @test4_as1(<4 x i32> %x, <4 x i32> %y, <4 x i32> addrspace(1)* %z) {
; CHECK-LABEL: @test4_as1(
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.y.cast = bitcast <4 x i32>* %a.y to i8*
  %z.cast = bitcast <4 x i32> addrspace(1)* %z to i8 addrspace(1)*
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* %a.y.cast, i8 addrspace(1)* %z.cast, i32 16, i32 1, i1 false)
; CHECK-NOT: memcpy

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %a.tmp1.cast = bitcast i32* %a.tmp1 to i8*
  %z.tmp1 = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %z, i16 0, i16 2
  %z.tmp1.cast = bitcast i32 addrspace(1)* %z.tmp1 to i8 addrspace(1)*
  call void @llvm.memcpy.p0i8.p1i8.i32(i8* %a.tmp1.cast, i8 addrspace(1)* %z.tmp1.cast, i32 4, i32 1, i1 false)
  %tmp1 = load i32, i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32, i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32, i32* %a.tmp3
; CHECK-NOT: memcpy
; CHECK:      %[[load:.*]] = load <4 x i32>, <4 x i32> addrspace(1)* %z
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds <4 x i32>, <4 x i32> addrspace(1)* %z, i64 0, i64 2
; CHECK-NEXT: %[[element_load:.*]] = load i32, i32 addrspace(1)* %[[gep]]
; CHECK-NEXT: %[[insert:.*]] = insertelement <4 x i32> %x, i32 %[[element_load]], i32 2
; CHECK-NEXT: extractelement <4 x i32> %[[insert]], i32 2
; CHECK-NEXT: extractelement <4 x i32> %[[load]], i32 3
; CHECK-NEXT: extractelement <4 x i32> %[[load]], i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

define i32 @test5(<4 x i32> %x, <4 x i32> %y, <4 x i32>* %z) {
; CHECK-LABEL: @test5(
; The same as the above, but with reversed source and destination for the
; element memcpy, and a self copy.
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.y.cast = bitcast <4 x i32>* %a.y to i8*
  %a.x.cast = bitcast <4 x i32>* %a.x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.x.cast, i8* %a.y.cast, i32 16, i32 1, i1 false)
; CHECK-NOT: memcpy

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %a.tmp1.cast = bitcast i32* %a.tmp1 to i8*
  %z.tmp1 = getelementptr inbounds <4 x i32>, <4 x i32>* %z, i64 0, i64 2
  %z.tmp1.cast = bitcast i32* %z.tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %z.tmp1.cast, i8* %a.tmp1.cast, i32 4, i32 1, i1 false)
  %tmp1 = load i32, i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32, i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>], [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32, i32* %a.tmp3
; CHECK-NOT: memcpy
; CHECK:      %[[gep:.*]] = getelementptr inbounds <4 x i32>, <4 x i32>* %z, i64 0, i64 2
; CHECK-NEXT: %[[extract:.*]] = extractelement <4 x i32> %y, i32 2
; CHECK-NEXT: store i32 %[[extract]], i32* %[[gep]]
; CHECK-NEXT: extractelement <4 x i32> %y, i32 2
; CHECK-NEXT: extractelement <4 x i32> %y, i32 3
; CHECK-NEXT: extractelement <4 x i32> %y, i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i32, i1) nounwind

define i64 @test6(<4 x i64> %x, <4 x i64> %y, i64 %n) {
; CHECK-LABEL: @test6(
; The old scalarrepl pass would wrongly drop the store to the second alloca.
; PR13254
  %tmp = alloca { <4 x i64>, <4 x i64> }
  %p0 = getelementptr inbounds { <4 x i64>, <4 x i64> }, { <4 x i64>, <4 x i64> }* %tmp, i32 0, i32 0
  store <4 x i64> %x, <4 x i64>* %p0
; CHECK: store <4 x i64> %x,
  %p1 = getelementptr inbounds { <4 x i64>, <4 x i64> }, { <4 x i64>, <4 x i64> }* %tmp, i32 0, i32 1
  store <4 x i64> %y, <4 x i64>* %p1
; CHECK: store <4 x i64> %y,
  %addr = getelementptr inbounds { <4 x i64>, <4 x i64> }, { <4 x i64>, <4 x i64> }* %tmp, i32 0, i32 0, i64 %n
  %res = load i64, i64* %addr, align 4
  ret i64 %res
}

define <4 x i32> @test_subvec_store() {
; CHECK-LABEL: @test_subvec_store(
entry:
  %a = alloca <4 x i32>
; CHECK-NOT: alloca

  %a.gep0 = getelementptr <4 x i32>, <4 x i32>* %a, i32 0, i32 0
  %a.cast0 = bitcast i32* %a.gep0 to <2 x i32>*
  store <2 x i32> <i32 0, i32 0>, <2 x i32>* %a.cast0
; CHECK-NOT: store
; CHECK:     select <4 x i1> <i1 true, i1 true, i1 false, i1 false> 

  %a.gep1 = getelementptr <4 x i32>, <4 x i32>* %a, i32 0, i32 1
  %a.cast1 = bitcast i32* %a.gep1 to <2 x i32>*
  store <2 x i32> <i32 1, i32 1>, <2 x i32>* %a.cast1
; CHECK-NEXT: select <4 x i1> <i1 false, i1 true, i1 true, i1 false>

  %a.gep2 = getelementptr <4 x i32>, <4 x i32>* %a, i32 0, i32 2
  %a.cast2 = bitcast i32* %a.gep2 to <2 x i32>*
  store <2 x i32> <i32 2, i32 2>, <2 x i32>* %a.cast2
; CHECK-NEXT: select <4 x i1> <i1 false, i1 false, i1 true, i1 true>

  %a.gep3 = getelementptr <4 x i32>, <4 x i32>* %a, i32 0, i32 3
  store i32 3, i32* %a.gep3
; CHECK-NEXT: insertelement <4 x i32>

  %ret = load <4 x i32>, <4 x i32>* %a

  ret <4 x i32> %ret
; CHECK-NEXT: ret <4 x i32> 
}

define <4 x i32> @test_subvec_load() {
; CHECK-LABEL: @test_subvec_load(
entry:
  %a = alloca <4 x i32>
; CHECK-NOT: alloca
  store <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32>* %a
; CHECK-NOT: store

  %a.gep0 = getelementptr <4 x i32>, <4 x i32>* %a, i32 0, i32 0
  %a.cast0 = bitcast i32* %a.gep0 to <2 x i32>*
  %first = load <2 x i32>, <2 x i32>* %a.cast0
; CHECK-NOT: load
; CHECK:      %[[extract1:.*]] = shufflevector <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32> undef, <2 x i32> <i32 0, i32 1>

  %a.gep1 = getelementptr <4 x i32>, <4 x i32>* %a, i32 0, i32 1
  %a.cast1 = bitcast i32* %a.gep1 to <2 x i32>*
  %second = load <2 x i32>, <2 x i32>* %a.cast1
; CHECK-NEXT: %[[extract2:.*]] = shufflevector <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32> undef, <2 x i32> <i32 1, i32 2>

  %a.gep2 = getelementptr <4 x i32>, <4 x i32>* %a, i32 0, i32 2
  %a.cast2 = bitcast i32* %a.gep2 to <2 x i32>*
  %third = load <2 x i32>, <2 x i32>* %a.cast2
; CHECK-NEXT: %[[extract3:.*]] = shufflevector <4 x i32> <i32 0, i32 1, i32 2, i32 3>, <4 x i32> undef, <2 x i32> <i32 2, i32 3>

  %tmp = shufflevector <2 x i32> %first, <2 x i32> %second, <2 x i32> <i32 0, i32 2>
  %ret = shufflevector <2 x i32> %tmp, <2 x i32> %third, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
; CHECK-NEXT: %[[tmp:.*]] = shufflevector <2 x i32> %[[extract1]], <2 x i32> %[[extract2]], <2 x i32> <i32 0, i32 2>
; CHECK-NEXT: %[[ret:.*]] = shufflevector <2 x i32> %[[tmp]], <2 x i32> %[[extract3]], <4 x i32> <i32 0, i32 1, i32 2, i32 3>

  ret <4 x i32> %ret
; CHECK-NEXT: ret <4 x i32> %[[ret]]
}

declare void @llvm.memset.p0i32.i32(i32* nocapture, i32, i32, i32, i1) nounwind

define <4 x float> @test_subvec_memset() {
; CHECK-LABEL: @test_subvec_memset(
entry:
  %a = alloca <4 x float>
; CHECK-NOT: alloca

  %a.gep0 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 0
  %a.cast0 = bitcast float* %a.gep0 to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.cast0, i8 0, i32 8, i32 0, i1 false)
; CHECK-NOT: store
; CHECK: select <4 x i1> <i1 true, i1 true, i1 false, i1 false>

  %a.gep1 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 1
  %a.cast1 = bitcast float* %a.gep1 to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.cast1, i8 1, i32 8, i32 0, i1 false)
; CHECK-NEXT: select <4 x i1> <i1 false, i1 true, i1 true, i1 false>

  %a.gep2 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 2
  %a.cast2 = bitcast float* %a.gep2 to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.cast2, i8 3, i32 8, i32 0, i1 false)
; CHECK-NEXT: select <4 x i1> <i1 false, i1 false, i1 true, i1 true>

  %a.gep3 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 3
  %a.cast3 = bitcast float* %a.gep3 to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.cast3, i8 7, i32 4, i32 0, i1 false)
; CHECK-NEXT: insertelement <4 x float> 

  %ret = load <4 x float>, <4 x float>* %a

  ret <4 x float> %ret
; CHECK-NEXT: ret <4 x float> 
}

define <4 x float> @test_subvec_memcpy(i8* %x, i8* %y, i8* %z, i8* %f, i8* %out) {
; CHECK-LABEL: @test_subvec_memcpy(
entry:
  %a = alloca <4 x float>
; CHECK-NOT: alloca

  %a.gep0 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 0
  %a.cast0 = bitcast float* %a.gep0 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.cast0, i8* %x, i32 8, i32 0, i1 false)
; CHECK:      %[[xptr:.*]] = bitcast i8* %x to <2 x float>*
; CHECK-NEXT: %[[x:.*]] = load <2 x float>, <2 x float>* %[[xptr]]
; CHECK-NEXT: %[[expand_x:.*]] = shufflevector <2 x float> %[[x]], <2 x float> undef, <4 x i32> <i32 0, i32 1, i32 undef, i32 undef>
; CHECK-NEXT: select <4 x i1> <i1 true, i1 true, i1 false, i1 false>  

  %a.gep1 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 1
  %a.cast1 = bitcast float* %a.gep1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.cast1, i8* %y, i32 8, i32 0, i1 false)
; CHECK-NEXT: %[[yptr:.*]] = bitcast i8* %y to <2 x float>*
; CHECK-NEXT: %[[y:.*]] = load <2 x float>, <2 x float>* %[[yptr]]
; CHECK-NEXT: %[[expand_y:.*]] = shufflevector <2 x float> %[[y]], <2 x float> undef, <4 x i32> <i32 undef, i32 0, i32 1, i32 undef>
; CHECK-NEXT: select <4 x i1> <i1 false, i1 true, i1 true, i1 false>

  %a.gep2 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 2
  %a.cast2 = bitcast float* %a.gep2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.cast2, i8* %z, i32 8, i32 0, i1 false)
; CHECK-NEXT: %[[zptr:.*]] = bitcast i8* %z to <2 x float>*
; CHECK-NEXT: %[[z:.*]] = load <2 x float>, <2 x float>* %[[zptr]]
; CHECK-NEXT: %[[expand_z:.*]] = shufflevector <2 x float> %[[z]], <2 x float> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
; CHECK-NEXT: select <4 x i1> <i1 false, i1 false, i1 true, i1 true>

  %a.gep3 = getelementptr <4 x float>, <4 x float>* %a, i32 0, i32 3
  %a.cast3 = bitcast float* %a.gep3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.cast3, i8* %f, i32 4, i32 0, i1 false)
; CHECK-NEXT: %[[fptr:.*]] = bitcast i8* %f to float*
; CHECK-NEXT: %[[f:.*]] = load float, float* %[[fptr]]
; CHECK-NEXT: %[[insert_f:.*]] = insertelement <4 x float> 

  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %out, i8* %a.cast2, i32 8, i32 0, i1 false)
; CHECK-NEXT: %[[outptr:.*]] = bitcast i8* %out to <2 x float>*
; CHECK-NEXT: %[[extract_out:.*]] = shufflevector <4 x float> %[[insert_f]], <4 x float> undef, <2 x i32> <i32 2, i32 3>
; CHECK-NEXT: store <2 x float> %[[extract_out]], <2 x float>* %[[outptr]]

  %ret = load <4 x float>, <4 x float>* %a

  ret <4 x float> %ret
; CHECK-NEXT: ret <4 x float> %[[insert_f]]
}

define i32 @PR14212() {
; CHECK-LABEL: @PR14212(
; This caused a crash when "splitting" the load of the i32 in order to promote
; the store of <3 x i8> properly. Heavily reduced from an OpenCL test case.
entry:
  %retval = alloca <3 x i8>, align 4
; CHECK-NOT: alloca

  store <3 x i8> undef, <3 x i8>* %retval, align 4
  %cast = bitcast <3 x i8>* %retval to i32*
  %load = load i32, i32* %cast, align 4
  ret i32 %load
; CHECK: ret i32
}

define <2 x i8> @PR14349.1(i32 %x) {
; CHECK: @PR14349.1
; The first testcase for broken SROA rewriting of split integer loads and
; stores due to smaller vector loads and stores. This particular test ensures
; that we can rewrite a split store of an integer to a store of a vector.
entry:
  %a = alloca i32
; CHECK-NOT: alloca

  store i32 %x, i32* %a
; CHECK-NOT: store

  %cast = bitcast i32* %a to <2 x i8>*
  %vec = load <2 x i8>, <2 x i8>* %cast
; CHECK-NOT: load

  ret <2 x i8> %vec
; CHECK: %[[trunc:.*]] = trunc i32 %x to i16
; CHECK: %[[cast:.*]] = bitcast i16 %[[trunc]] to <2 x i8>
; CHECK: ret <2 x i8> %[[cast]]
}

define i32 @PR14349.2(<2 x i8> %x) {
; CHECK: @PR14349.2
; The first testcase for broken SROA rewriting of split integer loads and
; stores due to smaller vector loads and stores. This particular test ensures
; that we can rewrite a split load of an integer to a load of a vector.
entry:
  %a = alloca i32
; CHECK-NOT: alloca

  %cast = bitcast i32* %a to <2 x i8>*
  store <2 x i8> %x, <2 x i8>* %cast
; CHECK-NOT: store

  %int = load i32, i32* %a
; CHECK-NOT: load

  ret i32 %int
; CHECK: %[[cast:.*]] = bitcast <2 x i8> %x to i16
; CHECK: %[[trunc:.*]] = zext i16 %[[cast]] to i32
; CHECK: %[[insert:.*]] = or i32 %{{.*}}, %[[trunc]]
; CHECK: ret i32 %[[insert]]
}

define i32 @test7(<2 x i32> %x, <2 x i32> %y) {
; Test that we can promote to vectors when the alloca doesn't mention any vector types.
; CHECK-LABEL: @test7(
entry:
	%a = alloca [2 x i64]
  %a.cast = bitcast [2 x i64]* %a to [2 x <2 x i32>]*
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* %a.cast, i64 0, i64 0
  store <2 x i32> %x, <2 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* %a.cast, i64 0, i64 1
  store <2 x i32> %y, <2 x i32>* %a.y
; CHECK-NOT: store

  %a.tmp1 = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* %a.cast, i64 0, i64 0, i64 1
  %tmp1 = load i32, i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* %a.cast, i64 0, i64 1, i64 1
  %tmp2 = load i32, i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <2 x i32>], [2 x <2 x i32>]* %a.cast, i64 0, i64 1, i64 0
  %tmp3 = load i32, i32* %a.tmp3
; CHECK-NOT: load
; CHECK:      extractelement <2 x i32> %x, i32 1
; CHECK-NEXT: extractelement <2 x i32> %y, i32 1
; CHECK-NEXT: extractelement <2 x i32> %y, i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
; CHECK-NEXT: add
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

define i32 @test8(<2 x i32> %x) {
; Ensure that we can promote an alloca that doesn't mention a vector type based
; on a single store with a vector type.
; CHECK-LABEL: @test8(
entry:
	%a = alloca i64
  %a.vec = bitcast i64* %a to <2 x i32>*
  %a.i32 = bitcast i64* %a to i32*
; CHECK-NOT: alloca

  store <2 x i32> %x, <2 x i32>* %a.vec
; CHECK-NOT: store

  %tmp1 = load i32, i32* %a.i32
  %a.tmp2 = getelementptr inbounds i32, i32* %a.i32, i64 1
  %tmp2 = load i32, i32* %a.tmp2
; CHECK-NOT: load
; CHECK:      extractelement <2 x i32> %x, i32 0
; CHECK-NEXT: extractelement <2 x i32> %x, i32 1

  %tmp4 = add i32 %tmp1, %tmp2
  ret i32 %tmp4
; CHECK-NEXT: add
; CHECK-NEXT: ret
}

define <2 x i32> @test9(i32 %x, i32 %y) {
; Ensure that we can promote an alloca that doesn't mention a vector type based
; on a single load with a vector type.
; CHECK-LABEL: @test9(
entry:
	%a = alloca i64
  %a.vec = bitcast i64* %a to <2 x i32>*
  %a.i32 = bitcast i64* %a to i32*
; CHECK-NOT: alloca

  store i32 %x, i32* %a.i32
  %a.tmp2 = getelementptr inbounds i32, i32* %a.i32, i64 1
  store i32 %y, i32* %a.tmp2
; CHECK-NOT: store
; CHECK:      %[[V1:.*]] = insertelement <2 x i32> undef, i32 %x, i32 0
; CHECK-NEXT: %[[V2:.*]] = insertelement <2 x i32> %[[V1]], i32 %y, i32 1

  %result = load <2 x i32>, <2 x i32>* %a.vec
; CHECK-NOT:  load

  ret <2 x i32> %result
; CHECK-NEXT: ret <2 x i32> %[[V2]]
}

define <2 x i32> @test10(<4 x i16> %x, i32 %y) {
; If there are multiple different vector types used, we should select the one
; with the widest elements.
; CHECK-LABEL: @test10(
entry:
	%a = alloca i64
  %a.vec1 = bitcast i64* %a to <2 x i32>*
  %a.vec2 = bitcast i64* %a to <4 x i16>*
  %a.i32 = bitcast i64* %a to i32*
; CHECK-NOT: alloca

  store <4 x i16> %x, <4 x i16>* %a.vec2
  %a.tmp2 = getelementptr inbounds i32, i32* %a.i32, i64 1
  store i32 %y, i32* %a.tmp2
; CHECK-NOT: store
; CHECK:      %[[V1:.*]] = bitcast <4 x i16> %x to <2 x i32>
; CHECK-NEXT: %[[V2:.*]] = insertelement <2 x i32> %[[V1]], i32 %y, i32 1

  %result = load <2 x i32>, <2 x i32>* %a.vec1
; CHECK-NOT:  load

  ret <2 x i32> %result
; CHECK-NEXT: ret <2 x i32> %[[V2]]
}

define <2 x float> @test11(<4 x i16> %x, i32 %y) {
; If there are multiple different element types for different vector types,
; pick the integer types. This isn't really important, but seems like the best
; heuristic for making a deterministic decision.
; CHECK-LABEL: @test11(
entry:
	%a = alloca i64
  %a.vec1 = bitcast i64* %a to <2 x float>*
  %a.vec2 = bitcast i64* %a to <4 x i16>*
  %a.i32 = bitcast i64* %a to i32*
; CHECK-NOT: alloca

  store <4 x i16> %x, <4 x i16>* %a.vec2
  %a.tmp2 = getelementptr inbounds i32, i32* %a.i32, i64 1
  store i32 %y, i32* %a.tmp2
; CHECK-NOT: store
; CHECK:      %[[V1:.*]] = bitcast i32 %y to <2 x i16>
; CHECK-NEXT: %[[V2:.*]] = shufflevector <2 x i16> %[[V1]], <2 x i16> undef, <4 x i32> <i32 undef, i32 undef, i32 0, i32 1>
; CHECK-NEXT: %[[V3:.*]] = select <4 x i1> <i1 false, i1 false, i1 true, i1 true>, <4 x i16> %[[V2]], <4 x i16> %x
; CHECK-NEXT: %[[V4:.*]] = bitcast <4 x i16> %[[V3]] to <2 x float>

  %result = load <2 x float>, <2 x float>* %a.vec1
; CHECK-NOT:  load

  ret <2 x float> %result
; CHECK-NEXT: ret <2 x float> %[[V4]]
}

define <4 x float> @test12() {
; CHECK-LABEL: @test12(
  %a = alloca <3 x i32>, align 16
; CHECK-NOT: alloca

  %cast1 = bitcast <3 x i32>* %a to <4 x i32>*
  store <4 x i32> undef, <4 x i32>* %cast1, align 16
; CHECK-NOT: store

  %cast2 = bitcast <3 x i32>* %a to <3 x float>*
  %cast3 = bitcast <3 x float>* %cast2 to <4 x float>*
  %vec = load <4 x float>, <4 x float>* %cast3
; CHECK-NOT: load

; CHECK:      %[[ret:.*]] = bitcast <4 x i32> undef to <4 x float>
; CHECK-NEXT: ret <4 x float> %[[ret]]
  ret <4 x float> %vec
}
