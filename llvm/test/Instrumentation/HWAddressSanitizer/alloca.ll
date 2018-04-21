; Test basic address sanitizer instrumentation.
;
; RUN: opt < %s -hwasan -S | FileCheck %s --check-prefixes=CHECK,DYNAMIC-SHADOW
; RUN: opt < %s -hwasan -hwasan-mapping-offset=0 -S | FileCheck %s --check-prefixes=CHECK,ZERO-BASED-SHADOW
; RUN: opt < %s -hwasan -hwasan-generate-tags-with-calls -S | FileCheck %s --check-prefix=WITH-CALLS

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-android"

declare void @use32(i32*)

define void @test_alloca() sanitize_hwaddress {
; CHECK-LABEL: @test_alloca(
; CHECK: %[[FP:[^ ]*]] = call i8* @llvm.frameaddress(i32 0)
; CHECK: %[[A:[^ ]*]] = ptrtoint i8* %[[FP]] to i64
; CHECK: %[[B:[^ ]*]] = lshr i64 %[[A]], 20
; CHECK: %[[BASE_TAG:[^ ]*]] = xor i64 %[[A]], %[[B]]

; CHECK: %[[X:[^ ]*]] = alloca i32, align 16
; CHECK: %[[X_TAG:[^ ]*]] = xor i64 %[[BASE_TAG]], 0
; CHECK: %[[X1:[^ ]*]] = ptrtoint i32* %[[X]] to i64
; CHECK: %[[C:[^ ]*]] = shl i64 %[[X_TAG]], 56
; CHECK: %[[D:[^ ]*]] = or i64 %[[X1]], %[[C]]
; CHECK: %[[X_HWASAN:[^ ]*]] = inttoptr i64 %[[D]] to i32*

; CHECK: %[[X_TAG2:[^ ]*]] = trunc i64 %[[X_TAG]] to i8
; CHECK: %[[E:[^ ]*]] = ptrtoint i32* %[[X]] to i64
; CHECK: %[[F:[^ ]*]] = lshr i64 %[[E]], 4
; DYNAMIC-SHADOW: %[[F_DYN:[^ ]*]] = add i64 %[[F]], %.hwasan.shadow
; DYNAMIC-SHADOW: %[[X_SHADOW:[^ ]*]] = inttoptr i64 %[[F_DYN]] to i8*
; ZERO-BASED-SHADOW: %[[X_SHADOW:[^ ]*]] = inttoptr i64 %[[F]] to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 1 %[[X_SHADOW]], i8 %[[X_TAG2]], i64 1, i1 false)
; CHECK: call void @use32(i32* nonnull %[[X_HWASAN]])

; CHECK: %[[X_TAG_UAR:[^ ]*]] = xor i64 %[[BASE_TAG]], 255
; CHECK: %[[X_TAG_UAR2:[^ ]*]] = trunc i64 %[[X_TAG_UAR]] to i8
; CHECK: %[[E2:[^ ]*]] = ptrtoint i32* %[[X]] to i64
; CHECK: %[[F2:[^ ]*]] = lshr i64 %[[E2]], 4
; DYNAMIC-SHADOW: %[[F2_DYN:[^ ]*]] = add i64 %[[F2]], %.hwasan.shadow
; DYNAMIC-SHADOW: %[[X_SHADOW2:[^ ]*]] = inttoptr i64 %[[F2_DYN]] to i8*
; ZERO-BASED-SHADOW: %[[X_SHADOW2:[^ ]*]] = inttoptr i64 %[[F2]] to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 1 %[[X_SHADOW2]], i8 %[[X_TAG_UAR2]], i64 1, i1 false)
; CHECK: ret void


entry:
  %x = alloca i32, align 4
  call void @use32(i32* nonnull %x)
  ret void
}

; WITH-CALLS-LABEL: @test_alloca(
; WITH-CALLS: %[[T1:[^ ]*]] = call i8 @__hwasan_generate_tag()
; WITH-CALLS: %[[A:[^ ]*]] = zext i8 %[[T1]] to i64
; WITH-CALLS: %[[B:[^ ]*]] = ptrtoint i32* %x to i64
; WITH-CALLS: %[[C:[^ ]*]] = shl i64 %[[A]], 56
; WITH-CALLS: or i64 %[[B]], %[[C]]
