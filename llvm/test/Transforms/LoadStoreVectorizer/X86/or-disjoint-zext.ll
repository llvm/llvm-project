; RUN: opt %s -mtriple=x86_64-unknown-linux-gnu -passes=load-store-vectorizer -S -o - | FileCheck %s
;
; Test that LoadStoreVectorizer handles `or disjoint` in GEP index
; computations, vectorizing loads with zext(or disjoint %idx, K) indices.

target datalayout = "e-p:64:64-p1:64:64-i64:64-n32:64"

; Case A (baseline): zext(add nuw) - vectorizes on x86 as <3 x i32> + scalar.
; CHECK-LABEL: @test_zext_add
; CHECK: load <3 x i32>
; CHECK: load i32
define void @test_zext_add(ptr %base, i32 %idx) {
  %i0 = zext i32 %idx to i64
  %g0 = getelementptr inbounds i32, ptr %base, i64 %i0
  %v0 = load i32, ptr %g0, align 4

  %idx1 = add nuw i32 %idx, 1
  %i1 = zext i32 %idx1 to i64
  %g1 = getelementptr inbounds i32, ptr %base, i64 %i1
  %v1 = load i32, ptr %g1, align 4

  %idx2 = add nuw i32 %idx, 2
  %i2 = zext i32 %idx2 to i64
  %g2 = getelementptr inbounds i32, ptr %base, i64 %i2
  %v2 = load i32, ptr %g2, align 4

  %idx3 = add nuw i32 %idx, 3
  %i3 = zext i32 %idx3 to i64
  %g3 = getelementptr inbounds i32, ptr %base, i64 %i3
  %v3 = load i32, ptr %g3, align 4

  %s = add i32 %v0, %v1
  %s2 = add i32 %s, %v2
  %s3 = add i32 %s2, %v3
  %out_idx = sext i32 %s3 to i64
  %out = getelementptr inbounds i32, ptr %base, i64 %out_idx
  store i32 %s3, ptr %out, align 4
  ret void
}

; Case B: zext(or disjoint) - same x86 vectorization shape as add nuw.
; CHECK-LABEL: @test_zext_or_disjoint
; CHECK: load <3 x i32>
; CHECK: load i32
define void @test_zext_or_disjoint(ptr %base, i32 %idx) {
  %i0 = zext i32 %idx to i64
  %g0 = getelementptr inbounds i32, ptr %base, i64 %i0
  %v0 = load i32, ptr %g0, align 4

  %b1 = or disjoint i32 %idx, 1
  %i1 = zext i32 %b1 to i64
  %g1 = getelementptr inbounds i32, ptr %base, i64 %i1
  %v1 = load i32, ptr %g1, align 4

  %b2 = or disjoint i32 %idx, 2
  %i2 = zext i32 %b2 to i64
  %g2 = getelementptr inbounds i32, ptr %base, i64 %i2
  %v2 = load i32, ptr %g2, align 4

  %b3 = or disjoint i32 %idx, 3
  %i3 = zext i32 %b3 to i64
  %g3 = getelementptr inbounds i32, ptr %base, i64 %i3
  %v3 = load i32, ptr %g3, align 4

  %s = add i32 %v0, %v1
  %s2 = add i32 %s, %v2
  %s3 = add i32 %s2, %v3
  %out_idx = sext i32 %s3 to i64
  %out = getelementptr inbounds i32, ptr %base, i64 %out_idx
  store i32 %s3, ptr %out, align 4
  ret void
}

; Plain or (not disjoint) - should NOT vectorize.
; CHECK-LABEL: @test_zext_or_plain
; CHECK-NOT: load <3 x i32>
; CHECK: load i32
define void @test_zext_or_plain(ptr %base, i32 %idx) {
  %i0 = zext i32 %idx to i64
  %g0 = getelementptr inbounds i32, ptr %base, i64 %i0
  %v0 = load i32, ptr %g0, align 4

  %b1 = or i32 %idx, 1
  %i1 = zext i32 %b1 to i64
  %g1 = getelementptr inbounds i32, ptr %base, i64 %i1
  %v1 = load i32, ptr %g1, align 4

  %b2 = or i32 %idx, 2
  %i2 = zext i32 %b2 to i64
  %g2 = getelementptr inbounds i32, ptr %base, i64 %i2
  %v2 = load i32, ptr %g2, align 4

  %b3 = or i32 %idx, 3
  %i3 = zext i32 %b3 to i64
  %g3 = getelementptr inbounds i32, ptr %base, i64 %i3
  %v3 = load i32, ptr %g3, align 4

  %s = add i32 %v0, %v1
  %s2 = add i32 %s, %v2
  %s3 = add i32 %s2, %v3
  %out_idx = sext i32 %s3 to i64
  %out = getelementptr inbounds i32, ptr %base, i64 %out_idx
  store i32 %s3, ptr %out, align 4
  ret void
}
