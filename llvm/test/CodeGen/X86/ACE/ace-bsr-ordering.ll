; RUN: opt -O2 -S < %s | FileCheck %s
; Verify that BSR SSA threading prevents reordering of BSR write relative to TOP4MX compute.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The x86_bsr argument creates a data dependency: the bsr_create result flows into
; the TOP4MX call, so the optimizer cannot move bsr_create past the compute.

define void @bsr_ordering_preserved(ptr %tile_ptr, ptr %zmm_ptr, ptr %bsr_ptr, i16 %m, i16 %n, i16 %k) #0 {
; CHECK-LABEL: @bsr_ordering_preserved
; Verify that bsr.create dominates top4mxhf8ps.bsr.internal (SSA guarantees this)
; CHECK: %bsr = {{.*}}call x86_bsr @llvm.x86.bsr.create(
; CHECK: {{.*}}call x86_amx @llvm.x86.top4mxhf8ps.bsr.internal({{.*}}, x86_bsr %bsr)
entry:
  %acc = call x86_amx @llvm.x86.tileloadd64.internal(i16 %m, i16 %n, ptr %tile_ptr, i64 64)
  %zmm1 = load <16 x i32>, ptr %zmm_ptr, align 64
  %bsr_lo = load <16 x i32>, ptr %bsr_ptr, align 64
  %bsr_hi_ptr = getelementptr inbounds <16 x i32>, ptr %bsr_ptr, i64 1
  %bsr_hi = load <16 x i32>, ptr %bsr_hi_ptr, align 64
  %bsr = call x86_bsr @llvm.x86.bsr.create(<16 x i32> %bsr_lo, <16 x i32> %bsr_hi)
  %result = call x86_amx @llvm.x86.top4mxhf8ps.bsr.internal(i16 %m, i16 %n, i16 %k, i8 0, x86_amx %acc, <16 x i32> %zmm1, <16 x i32> %zmm1, x86_bsr %bsr)
  call void @llvm.x86.tilestored64.internal(i16 %m, i16 %n, ptr %tile_ptr, i64 64, x86_amx %result)
  ret void
}

attributes #0 = { nounwind "target-features"="+ace,+avx512f,+amx-tile" }

declare x86_amx @llvm.x86.tileloadd64.internal(i16, i16, ptr, i64)
declare void @llvm.x86.tilestored64.internal(i16, i16, ptr, i64, x86_amx)
declare x86_bsr @llvm.x86.bsr.create(<16 x i32>, <16 x i32>)
declare x86_amx @llvm.x86.top4mxhf8ps.bsr.internal(i16, i16, i16, i8, x86_amx, <16 x i32>, <16 x i32>, x86_bsr)
