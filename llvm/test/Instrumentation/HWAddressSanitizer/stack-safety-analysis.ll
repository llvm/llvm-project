; RUN: opt -passes=hwasan -hwasan-instrument-with-calls -hwasan-use-stack-safety=1 -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=SAFETY,CHECK
; RUN: opt -passes=hwasan -hwasan-instrument-with-calls -hwasan-use-stack-safety=0 -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=NOSAFETY,CHECK
; RUN: opt -passes=hwasan -hwasan-instrument-with-calls -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=SAFETY,CHECK
; RUN: opt -passes=hwasan -hwasan-instrument-stack=0 -hwasan-instrument-with-calls -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=NOSTACK,CHECK

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Check a safe alloca to ensure it does not get a tag.
define i32 @test_simple(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_simple
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %buf.sroa.0)
  store volatile i8 0, ptr %buf.sroa.0, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %buf.sroa.0)
  ret i32 0
}

; Check a non-safe alloca to ensure it gets a tag.
define i32 @test_use(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_use
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  call void @use(ptr nonnull %buf.sroa.0)
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %buf.sroa.0)
  store volatile i8 0, ptr %buf.sroa.0, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %buf.sroa.0)
  ret i32 0
}

; Check an alloca with in range GEP to ensure it does not get a tag or check.
define i32 @test_in_range(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca [10 x i8], align 4
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %buf.sroa.0)
  store volatile i8 0, ptr %buf.sroa.0, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %buf.sroa.0)
  ret i32 0
}

; Check an alloca with in range GEP to ensure it does not get a tag or check.
define i32 @test_in_range2(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range2
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %buf.sroa.0)
  store volatile i8 0, ptr %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %buf.sroa.0)
  ret i32 0
}

define i32 @test_in_range3(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range3
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memset
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_memset
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memset
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.memset.p0.i32(ptr %ptr, i8 0, i32 1, i1 true)
  ret i32 0
}

define i32 @test_in_range4(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range4
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.memmove.p0.p0.i32(ptr %ptr, ptr %ptr, i32 1, i1 true)
  ret i32 0
}

define i32 @test_in_range5(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_in_range5
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  %buf.sroa.1 = alloca [10 x i8], align 4
  %ptr1 = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.memmove.p0.p0.i32(ptr %ptr, ptr %ptr1, i32 1, i1 true)
  ret i32 0
}

; Check an alloca with out of range GEP to ensure it gets a tag and check.
define i32 @test_out_of_range(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 10
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %buf.sroa.0)
  store volatile i8 0, ptr %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %buf.sroa.0)
  ret i32 0
}

define i32 @test_out_of_range3(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range3
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memset
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memset
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memset
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.memset.p0.i32(ptr %ptr, i8 0, i32 2, i1 true)
  ret i32 0
}

define i32 @test_out_of_range4(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range4
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.memmove.p0.p0.i32(ptr %ptr, ptr %ptr, i32 2, i1 true)
  ret i32 0
}

define i32 @test_out_of_range5(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_out_of_range5
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  %buf.sroa.1 = alloca [10 x i8], align 4
  %ptr1 = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %buf.sroa.0)
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %buf.sroa.0)
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %buf.sroa.1)
  call void @llvm.memmove.p0.p0.i32(ptr %ptr, ptr %ptr1, i32 1, i1 true)
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %buf.sroa.1)
  ret i32 0
}

; Check an alloca with potentially out of range GEP to ensure it gets a tag and
; check.
define i32 @test_potentially_out_of_range(ptr %a) sanitize_hwaddress {
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
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 %off
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %ptr)
  store volatile i8 0, ptr %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %ptr)
  ret i32 0
}

define i32 @test_potentially_out_of_range2(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_potentially_out_of_range2
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_memmove
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_memmove
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK: call {{.*}}__hwasan_memmove
  %buf.sroa.0 = alloca [10 x i8], align 4
  %ptr = getelementptr [10 x i8], ptr %buf.sroa.0, i32 0, i32 9
  call void @llvm.memmove.p0.p0.i32(ptr %ptr, ptr %a, i32 1, i1 true)
  ret i32 0
}
; Check an alloca with potentially out of range GEP to ensure it gets a tag and
; check.
define i32 @test_unclear(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_unclear
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  %ptr = call ptr @getptr(ptr %buf.sroa.0)
  call void @llvm.lifetime.start.p0(i64 10, ptr nonnull %ptr)
  store volatile i8 0, ptr %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 10, ptr nonnull %ptr)
  ret i32 0
}

define i32 @test_select(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_select
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK: call {{.*}}__hwasan_store
  %x = call ptr @getptr(ptr %a)
  %buf.sroa.0 = alloca i8, align 4
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %buf.sroa.0)
  %c = call i1 @cond()
  %ptr = select i1 %c, ptr %x, ptr %buf.sroa.0
  store volatile i8 0, ptr %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %buf.sroa.0)
  ret i32 0
}

; Check whether we see through the returns attribute of functions.
define i32 @test_retptr(ptr %a) sanitize_hwaddress {
entry:
  ; CHECK-LABEL: @test_retptr
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; NOSAFETY: call {{.*}}__hwasan_store
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_store
  ; NOSTACK-NOT: call {{.*}}__hwasan_generate_tag
  ; NOSTACK-NOT: call {{.*}}__hwasan_store
  %buf.sroa.0 = alloca i8, align 4
  call void @llvm.lifetime.start.p0(i64 1, ptr nonnull %buf.sroa.0)
  %ptr = call ptr @retptr(ptr %buf.sroa.0)
  store volatile i8 0, ptr %ptr, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0(i64 1, ptr nonnull %buf.sroa.0)
  ret i32 0
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture)

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture)

declare void @llvm.memset.p0.i32(ptr, i8, i32, i1)
declare void @llvm.memcpy.p0.p0.i32(ptr, ptr, i32, i1)
declare void @llvm.memmove.p0.p0.i32(ptr, ptr, i32, i1)

declare i1 @cond()
declare void @use(ptr nocapture)
declare i32 @getoffset()
declare ptr @getptr(ptr nocapture)
declare ptr @retptr(ptr returned)

!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
