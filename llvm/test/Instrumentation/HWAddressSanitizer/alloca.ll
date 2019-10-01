; Test alloca instrumentation.
;
; RUN: opt < %s -hwasan -hwasan-with-ifunc=1 -S | FileCheck %s --check-prefixes=CHECK,DYNAMIC-SHADOW,NO-UAR-TAGS
; RUN: opt < %s -hwasan -hwasan-mapping-offset=0 -S | FileCheck %s --check-prefixes=CHECK,ZERO-BASED-SHADOW,NO-UAR-TAGS
; RUN: opt < %s -hwasan -hwasan-with-ifunc=1 -hwasan-uar-retag-to-zero=0 -S | FileCheck %s --check-prefixes=CHECK,DYNAMIC-SHADOW,UAR-TAGS

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
; DYNAMIC-SHADOW: %[[X_SHADOW:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %[[F]]
; ZERO-BASED-SHADOW: %[[X_SHADOW:[^ ]*]] = inttoptr i64 %[[F]] to i8*
; CHECK: call void @llvm.memset.p0i8.i64(i8* align 1 %[[X_SHADOW]], i8 %[[X_TAG2]], i64 1, i1 false)
; CHECK: call void @use32(i32* nonnull %[[X_HWASAN]])

; UAR-TAGS: %[[BASE_TAG_COMPL:[^ ]*]] = xor i64 %[[BASE_TAG]], 255
; UAR-TAGS: %[[X_TAG_UAR:[^ ]*]] = trunc i64 %[[BASE_TAG_COMPL]] to i8
; CHECK: %[[E2:[^ ]*]] = ptrtoint i32* %[[X]] to i64
; CHECK: %[[F2:[^ ]*]] = lshr i64 %[[E2]], 4
; DYNAMIC-SHADOW: %[[X_SHADOW2:[^ ]*]] = getelementptr i8, i8* %.hwasan.shadow, i64 %[[F2]]
; ZERO-BASED-SHADOW: %[[X_SHADOW2:[^ ]*]] = inttoptr i64 %[[F2]] to i8*
; NO-UAR-TAGS: call void @llvm.memset.p0i8.i64(i8* align 1 %[[X_SHADOW2]], i8 0, i64 1, i1 false)
; UAR-TAGS: call void @llvm.memset.p0i8.i64(i8* align 1 %[[X_SHADOW2]], i8 %[[X_TAG_UAR]], i64 1, i1 false)
; CHECK: ret void


entry:
  %x = alloca i32, align 4
  call void @use32(i32* nonnull %x)
  ret void
}
