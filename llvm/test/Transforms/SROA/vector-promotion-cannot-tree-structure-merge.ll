; REQUIRES: asserts
; RUN: opt < %s -passes='sroa<preserve-cfg>' -disable-output -debug-only=sroa 2>&1 | FileCheck %s
; RUN: opt < %s -passes='sroa<modify-cfg>' -disable-output -debug-only=sroa 2>&1 | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

; CHECK-NOT: Tree structured merge rewrite
define i32 @test_alloca_not_fixed_vector() {
entry:
  %alloca = alloca [4 x float]

  %ptr0 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 0
  store float 1.0, ptr %ptr0

  %ptr1 = getelementptr inbounds [4 x float], ptr %alloca, i32 0, i32 1
  store float 2.0, ptr %ptr1

  %result = load i32, ptr %alloca
  ret i32 %result
}

define <4 x float> @test_more_than_one_load(<2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0

  %ptr1 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 2
  store <2 x float> %b, ptr %ptr1

  %result1 = load <4 x float>, ptr %alloca
  %result2 = load <4 x float>, ptr %alloca

  %final = fadd <4 x float> %result1, %result2
  ret <4 x float> %final
}

define void @test_no_load(<4 x float> %a) {
entry:
  %alloca = alloca <4 x float>
  store <4 x float> %a, ptr %alloca
  ret void
}

define i32 @test_load_not_fixed_vector(<2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0

  %ptr1 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 2
  store <2 x float> %b, ptr %ptr1

  %result = load i32, ptr %alloca
  ret i32 %result
}

define <3 x float> @test_load_not_covering_alloca(<2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0

  %ptr1 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 2
  store <2 x float> %b, ptr %ptr1

  %result = load <3 x float>, ptr %ptr0
  ret <3 x float> %result
}

define <4 x float> @test_store_not_fixed_vector(<vscale x 2 x float> %a) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  %fixed = extractelement <vscale x 2 x float> %a, i32 0
  store float %fixed, ptr %ptr0

  %result = load <4 x float>, ptr %alloca
  ret <4 x float> %result
}


define <4 x float> @test_no_stores() {
entry:
  %alloca = alloca <4 x float>

  %result = load <4 x float>, ptr %alloca
  ret <4 x float> %result
}

define <4 x float> @test_stores_overlapping(<2 x float> %a, <2 x float> %b, <2 x float> %c) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0

  %ptr1 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 1
  store <2 x float> %b, ptr %ptr1

  %ptr2 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 2
  store <2 x float> %c, ptr %ptr2

  %result = load <4 x float>, ptr %alloca
  ret <4 x float> %result
}

define <4 x float> @test_stores_not_covering_alloca(<2 x float> %a) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0

  %result = load <4 x float>, ptr %alloca
  ret <4 x float> %result
}

define <4 x float> @test_stores_not_same_basic_block(<2 x float> %a, <2 x float> %b, i1 %cond) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0

  br i1 %cond, label %then, label %else

then:
  %ptr1 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 2
  store <2 x float> %b, ptr %ptr1
  br label %merge

else:
  br label %merge

merge:
  %result = load <4 x float>, ptr %alloca
  ret <4 x float> %result
}

define <4 x float> @test_load_before_stores(<2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca <4 x float>

  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0

  %intermediate = load <4 x float>, ptr %alloca

  %ptr1 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 2
  store <2 x float> %b, ptr %ptr1

  ret <4 x float> %intermediate
}

define <4 x float> @test_other_instructions(<2 x float> %a, <2 x float> %b) {
entry:
  %alloca = alloca <4 x float>
  
  ; Store first vector
  %ptr0 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 0
  store <2 x float> %a, ptr %ptr0
  
  ; Other instruction (memset) that's not a simple load/store
  call void @llvm.memset.p0.i64(ptr %alloca, i8 0, i64 8, i1 false)
  
  ; Store second vector
  %ptr1 = getelementptr inbounds <4 x float>, ptr %alloca, i32 0, i32 2
  store <2 x float> %b, ptr %ptr1
  
  %result = load <4 x float>, ptr %alloca
  ret <4 x float> %result
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
