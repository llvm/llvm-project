; RUN: opt -hwasan -hwasan-use-stack-safety=1 -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=SAFETY
; RUN: opt -hwasan -hwasan-use-stack-safety=0 -hwasan-generate-tags-with-calls -S < %s | FileCheck %s --check-prefixes=NOSAFETY

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; Check a safe alloca to ensure it does not get a tag.
define i32 @test_load(i32* %a) sanitize_hwaddress {
entry:
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY-NOT: call {{.*}}__hwasan_generate_tag
  %buf.sroa.0 = alloca i8, align 4
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  store volatile i8 0, i8* %buf.sroa.0, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  ret i32 0
}

; Check a non-safe alloca to ensure it gets a tag.
define i32 @test_use(i32* %a) sanitize_hwaddress {
entry:
  ; NOSAFETY: call {{.*}}__hwasan_generate_tag
  ; SAFETY: call {{.*}}__hwasan_generate_tag
  %buf.sroa.0 = alloca i8, align 4
  call void @use(i8* nonnull %buf.sroa.0)
  call void @llvm.lifetime.start.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  store volatile i8 0, i8* %buf.sroa.0, align 4, !tbaa !8
  call void @llvm.lifetime.end.p0i8(i64 1, i8* nonnull %buf.sroa.0)
  ret i32 0
}

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture)

; Function Attrs: argmemonly mustprogress nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture)

declare void @use(i8* nocapture)

!8 = !{!9, !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
