; RUN: opt -passes=hwasan -hwasan-instrument-with-calls -hwasan-use-stack-safety=1 -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=SAFETY,CHECK
; RUN: opt -passes=hwasan -hwasan-instrument-with-calls -hwasan-use-stack-safety=0 -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=NOSAFETY,CHECK
; RUN: opt -passes=hwasan -hwasan-instrument-with-calls -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=SAFETY,CHECK
; RUN: opt -passes=hwasan -hwasan-instrument-stack=0 -hwasan-instrument-with-calls -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=NOSTACK,CHECK

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Check a safe alloca to ensure it does not get a tag.
define i32 @test_simple(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_simple
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  store volatile i8 0, i8* %buf.sroa.0, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  ret i32 0
}

; Check a non-safe alloca to ensure it gets a tag.
define i32 @test_use(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_use
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  call void @use(i8* nonnull %buf.sroa.0)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  store volatile i8 0, i8* %buf.sroa.0, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  ret i32 0
}

; Check an alloca with in range GEP to ensure it does not get a tag or check.
define i32 @test_in_range(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 0
  call void @llvm.lifetime.start.p0i8(i64 10, i8* nonnull %ptr)
  store volatile i8 0, i8* %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 10, i8* nonnull %ptr)
  ret i32 0
}

; Check an alloca with in range GEP to ensure it does not get a tag or check.
define i32 @test_in_range2(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range2
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 10, i8* nonnull %x)
  store volatile i8 0, i8* %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 10, i8* nonnull %x)
  ret i32 0
}

define i32 @test_in_range3(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range3
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memset
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_memset
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memset
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 0, i32 1, i1 true)
  ret i32 0
}

define i32 @test_in_range4(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range4
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %ptr, i8* %ptr, i32 1, i1 true)
  ret i32 0
}

define i32 @test_in_range5(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range5
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  %buf.sroa.1 = alloca [10 x i8], align 4
  %ptr1 = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %y = bitcast [10 x i8]* %buf.sroa.1 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %ptr, i8* %ptr1, i32 1, i1 true)
  ret i32 0
}

; Check an alloca with out of range GEP to ensure it gets a tag and check.
define i32 @test_out_of_range(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 10
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 10, i8* nonnull %x)
  store volatile i8 0, i8* %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 10, i8* nonnull %x)
  ret i32 0
}

define i32 @test_out_of_range3(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range3
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memset
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memset
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memset
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 0, i32 2, i1 true)
  ret i32 0
}

define i32 @test_out_of_range4(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range4
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %ptr, i8* %ptr, i32 2, i1 true)
  ret i32 0
}

define i32 @test_out_of_range5(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range5
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  %buf.sroa.1 = alloca [10 x i8], align 4
  %ptr1 = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %y = bitcast [10 x i8]* %buf.sroa.1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 10, i8* nonnull %x)
  call void @llvm.lifetime.end.p0i8(i64 10, i8* nonnull %x)
  call void @llvm.lifetime.start.p0i8(i64 10, i8* nonnull %y)
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %ptr, i8* %ptr1, i32 1, i1 true)
  call void @llvm.lifetime.end.p0i8(i64 10, i8* nonnull %y)
  ret i32 0
}

; Check an alloca with potentially out of range GEP to ensure it gets a tag and
; check.
define i32 @test_potentially_out_of_range(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_potentially_out_of_range
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca [10 x i8], align 4
  %off = call i32 @getoffset()
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 %off
  call void @llvm.lifetime.start.p0i8(i64 10, i8* nonnull %ptr)
  store volatile i8 0, i8* %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 10, i8* nonnull %ptr)
  ret i32 0
}

define i32 @test_potentially_out_of_range2(i8* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_potentially_out_of_range2
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], [10 x i8]* %buf.sroa.0, i32 0, i32 9
  %x = bitcast [10 x i8]* %buf.sroa.0 to i8*
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %ptr, i8* %a, i32 1, i1 true)
  ret i32 0
}
; Check an alloca with potentially out of range GEP to ensure it gets a tag and
; check.
define i32 @test_unclear(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_unclear
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  %ptr = call i8* @getptr(i8* %buf.sroa.0)
  call void @llvm.lifetime.start.p0i8(i64 10, i8* nonnull %ptr)
  store volatile i8 0, i8* %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 10, i8* nonnull %ptr)
  ret i32 0
}

define i32 @test_select(i8* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_select
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK: call {{.*}}__hwasan_store
  %x = call i8* @getptr(i8* %a)
  %buf.sroa.0 = alloca i8, align 4
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  %c = call i1 @cond()
  %ptr = select i1 %c, i8* %x, i8* %buf.sroa.0
  store volatile i8 0, i8* %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  ret i32 0
}

; Check whether we see through the returns attribute of functions.
define i32 @test_retptr(i32* %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_retptr
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  %ptr = call i8* @retptr(i8* %buf.sroa.0)
  store volatile i8 0, i8* %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  ret i32 0
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

declare void @llvm.memset.p0i8.i32(i8*, i8, i32, i1)
declare void @llvm.memcpy.p0i8.p0i8.i32(i8*, i8*, i32, i1)
declare void @llvm.memmove.p0i8.p0i8.i32(i8*, i8*, i32, i1)

declare i1 @cond()
declare void @use(i8* nocapture)
declare i32 @getoffset()
declare i8* @getptr(i8* nocapture)
declare i8* @retptr(i8* returned)

!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
