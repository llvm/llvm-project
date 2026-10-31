; RUN: opt -passes='sroa<aggregate-to-vector>' -S %s | FileCheck %s
; NOTE: Do not autogenerate. This test intentionally uses targeted CHECK
; patterns for clarity.

; When SROA splits { ptr, i64, i64, i64 } into [0,8), [8,16), [16,32),
; the [16,32) partition type from getTypePartition is { i64, i64 }.
; With the simplified fallback rule, that partition canonicalizes to <2 x i64>
; because its remaining users are all mem intrinsics.

; CHECK-LABEL: define void @test_subpartition_type(
; CHECK: %a.sroa.6.0.copyload = load <2 x i64>, ptr %a.sroa.6.0.src.sroa_idx, align 8
; CHECK: store <2 x i64> %a.sroa.6.0.copyload, ptr %a.sroa.6.0.dst.sroa_idx, align 8
define void @test_subpartition_type(ptr %src, ptr %dst) {
entry:
  %a = alloca { ptr, i64, i64, i64 }, align 8
  call void @llvm.lifetime.start.p0(i64 32, ptr %a)

  ; Copy all 32 bytes from src into %a (splittable)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %a, ptr align 8 %src, i64 32, i1 false)

  ; Load ptr at [0,8) -- forces partition boundary at 8
  %p = load ptr, ptr %a, align 8

  ; Load i64 at [8,16) -- forces partition boundary at 16
  %gep.a.8 = getelementptr inbounds i8, ptr %a, i64 8
  %v1 = load i64, ptr %gep.a.8, align 8

  ; Only splittable memcpy uses touch [16,32), so SROA creates a single
  ; [16,32) partition. getTypePartition returns { i64, i64 } for this, and
  ; the simplified fallback rule canonicalizes it to <2 x i64>.

  ; Copy all 32 bytes from %a to dst (splittable)
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %a, i64 32, i1 false)

  call void @llvm.lifetime.end.p0(i64 32, ptr %a)
  ret void
}

; Element-wise { double, double } access through a phi.
; The phi between two allocas prevents SROA slice analysis
; ("A pointer to this alloca escaped"), so the allocas survive.

; CHECK-LABEL: define void @test_elementwise_phi(
; CHECK-NOT: <2 x double>
define void @test_elementwise_phi(ptr %src0, ptr %src1, i1 %cond, ptr %dst) {
entry:
  %a = alloca { double, double }, align 8
  %b = alloca { double, double }, align 8
  %a.1 = getelementptr inbounds i8, ptr %a, i64 8
  %b.1 = getelementptr inbounds i8, ptr %b, i64 8
  %v0 = load double, ptr %src0, align 8
  %v1 = load double, ptr %src1, align 8
  store double %v0, ptr %a, align 8
  store double %v1, ptr %a.1, align 8
  store double 0.0, ptr %b, align 8
  store double 0.0, ptr %b.1, align 8
  br i1 %cond, label %if.then, label %if.else

if.then:
  br label %merge

if.else:
  br label %merge

merge:
  %sel = phi ptr [ %a, %if.then ], [ %b, %if.else ]
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %sel, i64 16, i1 false)
  ret void
}

; SROA sees these slices:
;   [0,8)   load ptr
;   [0,32)  store i256, splittable
;   [8,16)  load i64, splittable
;   [16,32) memcpy source, splittable
;
; These form three partitions:
;   [0,8)   contains the ptr load and the store i256 slice that starts at 0
;   [8,16)  contains the i64 load, plus the store i256 split tail
;   [16,32) contains the memcpy source, plus the store i256 split tail
;
; The [16,32) subpartition has type { i64, i64 }, and the only slice that
; starts in the partition is a memcpy. However, the whole-alloca i256 store is a
; split tail overlapping the subpartition, so it must block struct-to-vector
; fallback canonicalization.

; CHECK-LABEL: define void @test_split_tail_store_blocks_subpartition_type(
; CHECK-NOT: <2 x i64>
; CHECK: [[TAIL_SHIFT:%.*]] = lshr i256 %x, 128
; CHECK-NOT: <2 x i64>
; CHECK: [[TAIL:%.*]] = trunc i256 [[TAIL_SHIFT]] to i128
; CHECK-NOT: <2 x i64>
; CHECK: store i128 [[TAIL]], ptr %dst, align 8
; CHECK-NOT: <2 x i64>
; CHECK: ret void
define void @test_split_tail_store_blocks_subpartition_type(ptr %dst, i256 %x) {
entry:
  %a = alloca { ptr, i64, i64, i64 }, align 8
  store i256 %x, ptr %a, align 8

  ; Force earlier partition boundaries so [16,32) is selected as a subpartition
  ; whose type from getTypePartition is { i64, i64 }.
  %p = load ptr, ptr %a, align 8
  %puse = ptrtoint ptr %p to i64
  %gep.a.8 = getelementptr inbounds i8, ptr %a, i64 8
  %v8 = load i64, ptr %gep.a.8, align 8
  %use = add i64 %puse, %v8
  store i64 %use, ptr %dst, align 8

  %gep.a.16 = getelementptr inbounds i8, ptr %a, i64 16
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %gep.a.16, i64 16, i1 false)
  ret void
}
