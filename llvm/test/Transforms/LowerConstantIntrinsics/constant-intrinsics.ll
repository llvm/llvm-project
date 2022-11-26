; RUN: opt -lower-constant-intrinsics -S < %s | FileCheck %s

;; Ensure that an unfoldable is.constant gets lowered reasonably in
;; optimized codegen, in particular, that the "true" branch is
;; eliminated.

;; Also ensure that any unfoldable objectsize is resolved in order.

;; CHECK-NOT: tail call i32 @subfun_1()
;; CHECK:     tail call i32 @subfun_2()
;; CHECK-NOT: tail call i32 @subfun_1()

declare i1 @llvm.is.constant.i32(i32 %a) nounwind readnone
declare i1 @llvm.is.constant.i64(i64 %a) nounwind readnone
declare i1 @llvm.is.constant.i256(i256 %a) nounwind readnone
declare i1 @llvm.is.constant.v2i64(<2 x i64> %a) nounwind readnone
declare i1 @llvm.is.constant.f32(float %a) nounwind readnone
declare i1 @llvm.is.constant.sl_i32i32s({i32, i32} %a) nounwind readnone
declare i1 @llvm.is.constant.a2i64([2 x i64] %a) nounwind readnone
declare i1 @llvm.is.constant.p0(ptr %a) nounwind readnone

declare i64 @llvm.objectsize.i64.p0(ptr, i1, i1, i1) nounwind readnone

declare i32 @subfun_1()
declare i32 @subfun_2()

define i32 @test_branch(i32 %in) nounwind {
  %v = call i1 @llvm.is.constant.i32(i32 %in)
  br i1 %v, label %True, label %False

True:
  %call1 = tail call i32 @subfun_1()
  ret i32 %call1

False:
  %call2 = tail call i32 @subfun_2()
  ret i32 %call2
}

;; llvm.objectsize is another tricky case which gets folded to -1 very
;; late in the game. We'd like to ensure that llvm.is.constant of
;; llvm.objectsize is true.
define i1 @test_objectsize(ptr %obj) nounwind {
;; CHECK-LABEL:    test_objectsize
;; CHECK-NOT:      llvm.objectsize
;; CHECK-NOT:      llvm.is.constant
;; CHECK:          ret i1 true
  %os = call i64 @llvm.objectsize.i64.p0(ptr %obj, i1 false, i1 false, i1 false)
  %os1 = add i64 %os, 1
  %v = call i1 @llvm.is.constant.i64(i64 %os1)
  ret i1 %v
}

@test_phi_a = dso_local global i32 0, align 4
declare dso_local i32 @test_phi_b(...)

; Function Attrs: nounwind uwtable
define dso_local i32 @test_phi() {
entry:
  %0 = load i32, ptr @test_phi_a, align 4
  %1 = tail call i1 @llvm.is.constant.i32(i32 %0)
  br i1 %1, label %cond.end, label %cond.false

cond.false:                                       ; preds = %entry
  %call = tail call i32 @test_phi_b() #3
  %.pre = load i32, ptr @test_phi_a, align 4
  br label %cond.end

cond.end:                                         ; preds = %entry, %cond.false
  %2 = phi i32 [ %.pre, %cond.false ], [ %0, %entry ]
  %cond = phi i32 [ %call, %cond.false ], [ 1, %entry ]
  %cmp = icmp eq i32 %cond, %2
  br i1 %cmp, label %cond.true1, label %cond.end4

cond.true1:                                       ; preds = %cond.end
  %call2 = tail call i32 @test_phi_b() #3
  br label %cond.end4

cond.end4:                                        ; preds = %cond.end, %cond.true1
  ret i32 undef
}

define i1 @test_various_types(i256 %int, float %float, <2 x i64> %vec, {i32, i32} %struct, [2 x i64] %arr, ptr %ptr) #0 {
; CHECK-LABEL: @test_various_types(
; CHECK-NOT: llvm.is.constant
  %v1 = call i1 @llvm.is.constant.i256(i256 %int)
  %v2 = call i1 @llvm.is.constant.f32(float %float)
  %v3 = call i1 @llvm.is.constant.v2i64(<2 x i64> %vec)
  %v4 = call i1 @llvm.is.constant.sl_i32i32s({i32, i32} %struct)
  %v5 = call i1 @llvm.is.constant.a2i64([2 x i64] %arr)
  %v6 = call i1 @llvm.is.constant.p0(ptr %ptr)

  %c1 = call i1 @llvm.is.constant.i256(i256 -1)
  %c2 = call i1 @llvm.is.constant.f32(float 17.0)
  %c3 = call i1 @llvm.is.constant.v2i64(<2 x i64> <i64 -1, i64 44>)
  %c4 = call i1 @llvm.is.constant.sl_i32i32s({i32, i32} {i32 -1, i32 32})
  %c5 = call i1 @llvm.is.constant.a2i64([2 x i64] [i64 -1, i64 32])
  %c6 = call i1 @llvm.is.constant.p0(ptr inttoptr (i32 42 to ptr))

  %x1 = add i1 %v1, %c1
  %x2 = add i1 %v2, %c2
  %x3 = add i1 %v3, %c3
  %x4 = add i1 %v4, %c4
  %x5 = add i1 %v5, %c5
  %x6 = add i1 %v6, %c6

  %res2 = add i1 %x1, %x2
  %res3 = add i1 %res2, %x3
  %res4 = add i1 %res3, %x4
  %res5 = add i1 %res4, %x5
  %res6 = add i1 %res5, %x6

  ret i1 %res6
}

@real_mode_blob_end = external dso_local global [0 x i8], align 1
define i1 @global_array() {
; CHECK-LABEL: @global_array(
; CHECK-NEXT: ret i1 false
  %1 = call i1 @llvm.is.constant.i64(i64 ptrtoint (ptr @real_mode_blob_end to i64))
  ret i1 %1
}
