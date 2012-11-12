; RUN: opt < %s -sroa -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

%S1 = type { i64, [42 x float] }

define i32 @test1(<4 x i32> %x, <4 x i32> %y) {
; CHECK: @test1
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %tmp1 = load i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32* %a.tmp3
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
; CHECK: @test2
; FIXME: This should be handled!
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK: alloca <4 x i32>

  %a.x = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %tmp1 = load i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %a.tmp3.cast = bitcast i32* %a.tmp3 to <2 x i32>*
  %tmp3.vec = load <2 x i32>* %a.tmp3.cast
  %tmp3 = extractelement <2 x i32> %tmp3.vec, i32 0

  %tmp4 = add i32 %tmp1, %tmp2
  %tmp5 = add i32 %tmp3, %tmp4
  ret i32 %tmp5
}

define i32 @test3(<4 x i32> %x, <4 x i32> %y) {
; CHECK: @test3
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.y.cast = bitcast <4 x i32>* %a.y to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.y.cast, i8 0, i32 16, i32 1, i1 false)
; CHECK-NOT: memset

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %a.tmp1.cast = bitcast i32* %a.tmp1 to i8*
  call void @llvm.memset.p0i8.i32(i8* %a.tmp1.cast, i8 -1, i32 4, i32 1, i1 false)
  %tmp1 = load i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32* %a.tmp3
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
; CHECK: @test4
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.y.cast = bitcast <4 x i32>* %a.y to i8*
  %z.cast = bitcast <4 x i32>* %z to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.y.cast, i8* %z.cast, i32 16, i32 1, i1 false)
; CHECK-NOT: memcpy

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %a.tmp1.cast = bitcast i32* %a.tmp1 to i8*
  %z.tmp1 = getelementptr inbounds <4 x i32>* %z, i64 0, i64 2
  %z.tmp1.cast = bitcast i32* %z.tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.tmp1.cast, i8* %z.tmp1.cast, i32 4, i32 1, i1 false)
  %tmp1 = load i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32* %a.tmp3
; CHECK-NOT: memcpy
; CHECK:      %[[load:.*]] = load <4 x i32>* %z
; CHECK-NEXT: %[[gep:.*]] = getelementptr inbounds <4 x i32>* %z, i64 0, i64 2
; CHECK-NEXT: %[[element_load:.*]] = load i32* %[[gep]]
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
; CHECK: @test5
; The same as the above, but with reversed source and destination for the
; element memcpy, and a self copy.
entry:
	%a = alloca [2 x <4 x i32>]
; CHECK-NOT: alloca

  %a.x = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0
  store <4 x i32> %x, <4 x i32>* %a.x
  %a.y = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1
  store <4 x i32> %y, <4 x i32>* %a.y
; CHECK-NOT: store

  %a.y.cast = bitcast <4 x i32>* %a.y to i8*
  %a.x.cast = bitcast <4 x i32>* %a.x to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %a.x.cast, i8* %a.y.cast, i32 16, i32 1, i1 false)
; CHECK-NOT: memcpy

  %a.tmp1 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 0, i64 2
  %a.tmp1.cast = bitcast i32* %a.tmp1 to i8*
  %z.tmp1 = getelementptr inbounds <4 x i32>* %z, i64 0, i64 2
  %z.tmp1.cast = bitcast i32* %z.tmp1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %z.tmp1.cast, i8* %a.tmp1.cast, i32 4, i32 1, i1 false)
  %tmp1 = load i32* %a.tmp1
  %a.tmp2 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 3
  %tmp2 = load i32* %a.tmp2
  %a.tmp3 = getelementptr inbounds [2 x <4 x i32>]* %a, i64 0, i64 1, i64 0
  %tmp3 = load i32* %a.tmp3
; CHECK-NOT: memcpy
; CHECK:      %[[gep:.*]] = getelementptr inbounds <4 x i32>* %z, i64 0, i64 2
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
; CHECK: @test6
; The old scalarrepl pass would wrongly drop the store to the second alloca.
; PR13254
  %tmp = alloca { <4 x i64>, <4 x i64> }
  %p0 = getelementptr inbounds { <4 x i64>, <4 x i64> }* %tmp, i32 0, i32 0
  store <4 x i64> %x, <4 x i64>* %p0
; CHECK: store <4 x i64> %x,
  %p1 = getelementptr inbounds { <4 x i64>, <4 x i64> }* %tmp, i32 0, i32 1
  store <4 x i64> %y, <4 x i64>* %p1
; CHECK: store <4 x i64> %y,
  %addr = getelementptr inbounds { <4 x i64>, <4 x i64> }* %tmp, i32 0, i32 0, i64 %n
  %res = load i64* %addr, align 4
  ret i64 %res
}

define i32 @PR14212() {
; CHECK: @PR14212
; This caused a crash when "splitting" the load of the i32 in order to promote
; the store of <3 x i8> properly. Heavily reduced from an OpenCL test case.
entry:
  %retval = alloca <3 x i8>, align 4
; CHECK-NOT: alloca

  store <3 x i8> undef, <3 x i8>* %retval, align 4
  %cast = bitcast <3 x i8>* %retval to i32*
  %load = load i32* %cast, align 4
  ret i32 %load
; CHECK: ret i32
}
