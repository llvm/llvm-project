; RUN: llc -mtriple=hexagon -mattr=+hvxv75,+hvx-length128b < %s | FileCheck %s

; Test v32i1 store operation
; CHECK-LABEL: test_v32i1_store:
; CHECK: memw(r{{[0-9]+}}+#0) = r{{[0-9]+}}
define void @test_v32i1_store(i32 %idx) {
entry:
  %ptr = tail call ptr @malloc(i64 1088)
  %aligned_ptr_int = ptrtoint ptr %ptr to i32
  %aligned_int = add i32 %aligned_ptr_int, 63
  %aligned_int_masked = and i32 %aligned_int, -64
  %aligned_ptr = inttoptr i32 %aligned_int_masked to ptr
  %shifted = shl i32 %idx, 3
  %cond = icmp slt i32 %shifted, 100
  %pred_elem = insertelement <32 x i1> poison, i1 %cond, i64 0
  %pred_broadcast = shufflevector <32 x i1> %pred_elem, <32 x i1> poison, <32 x i32> zeroinitializer
  %mask = and <32 x i1> %pred_broadcast, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>
  store <32 x i1> %mask, ptr %aligned_ptr, align 64
  ret void
}

; Test v32i1 load operation
; CHECK-LABEL: test_v32i1_load:
; CHECK: r{{[0-9]+}} = memw(r{{[0-9]+}}+#0)
define <32 x i1> @test_v32i1_load(ptr %ptr) {
entry:
  %pred = load <32 x i1>, ptr %ptr, align 64
  ret <32 x i1> %pred
}

; Test v64i1 store operation
; CHECK-LABEL: test_v64i1_store:
; CHECK: memd(r{{[0-9]+}}+#0) = r{{[0-9]+}}:{{[0-9]+}}
define void @test_v64i1_store(i32 %idx) {
entry:
  %ptr = tail call ptr @malloc(i64 1088)
  %aligned_ptr_int = ptrtoint ptr %ptr to i32
  %aligned_int = add i32 %aligned_ptr_int, 63
  %aligned_int_masked = and i32 %aligned_int, -64
  %aligned_ptr = inttoptr i32 %aligned_int_masked to ptr
  %shifted = shl i32 %idx, 3
  %cond = icmp slt i32 %shifted, 100
  %pred_elem = insertelement <64 x i1> poison, i1 %cond, i64 0
  %pred_broadcast = shufflevector <64 x i1> %pred_elem, <64 x i1> poison, <64 x i32> zeroinitializer
  %mask = and <64 x i1> %pred_broadcast, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>
  store <64 x i1> %mask, ptr %aligned_ptr, align 64
  ret void
}

; Test v64i1 load operation
; CHECK-LABEL: test_v64i1_load:
; CHECK: r{{[0-9]+}}:{{[0-9]+}} = memd(r{{[0-9]+}}+#0)
define <64 x i1> @test_v64i1_load(ptr %ptr) {
entry:
  %pred = load <64 x i1>, ptr %ptr, align 64
  ret <64 x i1> %pred
}

; Test v128i1 store operation
; CHECK-LABEL: test_v128i1_store:
; CHECK: memd(r{{[0-9]+}}+#0) = r{{[0-9]+}}:{{[0-9]+}}
; CHECK: memd(r{{[0-9]+}}+#{{[0-9]+}}) = r{{[0-9]+}}:{{[0-9]+}}
define void @test_v128i1_store(i32 %idx) {
entry:
  %ptr = tail call ptr @malloc(i64 1088)
  %aligned_ptr_int = ptrtoint ptr %ptr to i32
  %aligned_int = add i32 %aligned_ptr_int, 63
  %aligned_int_masked = and i32 %aligned_int, -64
  %aligned_ptr = inttoptr i32 %aligned_int_masked to ptr
  %shifted = shl i32 %idx, 3
  %cond = icmp slt i32 %shifted, 100
  %pred_elem = insertelement <128 x i1> poison, i1 %cond, i64 0
  %pred_broadcast = shufflevector <128 x i1> %pred_elem, <128 x i1> poison, <128 x i32> zeroinitializer
  %mask = and <128 x i1> %pred_broadcast, <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>
  store <128 x i1> %mask, ptr %aligned_ptr, align 64
  ret void
}

; Test v128i1 load operation
; CHECK-LABEL: test_v128i1_load:
; CHECK-DAG: r{{[0-9]+}}:{{[0-9]+}} = memd(r{{[0-9]+}}+#0)
; CHECK-DAG: r{{[0-9]+}}:{{[0-9]+}} = memd(r{{[0-9]+}}+#{{[0-9]+}})
define <128 x i1> @test_v128i1_load(ptr %ptr) {
entry:
  %pred = load <128 x i1>, ptr %ptr, align 64
  ret <128 x i1> %pred
}

declare ptr @malloc(i64)
