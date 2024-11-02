; RUN: llc -mtriple=arm64-apple-ios -global-isel -global-isel-abort=1 -verify-machineinstrs -stop-after=aarch64-prelegalizer-combiner -force-legal-indexing %s -o - | FileCheck %s
; RUN: llc -debugify-and-strip-all-safe -mtriple=arm64-apple-ios -global-isel -global-isel-abort=1 -verify-machineinstrs -stop-after=aarch64-prelegalizer-combiner -force-legal-indexing %s -o - | FileCheck %s

define ptr @test_simple_load_pre(ptr %ptr) {
; CHECK-LABEL: name: test_simple_load_pre
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: {{%.*}}:_(s8), [[NEXT:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 1
; CHECK: $x0 = COPY [[NEXT]](p0)

  %next = getelementptr i8, ptr %ptr, i32 42
  load volatile i8, ptr %next
  ret ptr %next
}

define ptr @test_unused_load_pre(ptr %ptr) {
; CHECK-LABEL: name: test_unused_load_pre
; CHECK-NOT: G_INDEXED_LOAD

  %next = getelementptr i8, ptr %ptr, i32 42
  load volatile i8, ptr %next
  ret ptr null
}

define void @test_load_multiple_dominated(ptr %ptr, i1 %tst, i1 %tst2) {
; CHECK-LABEL: name: test_load_multiple_dominated
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: {{%.*}}:_(s8), [[NEXT:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 1
; CHECK: $x0 = COPY [[NEXT]](p0)
  %next = getelementptr i8, ptr %ptr, i32 42
  br i1 %tst, label %do_load, label %end

do_load:
  load volatile i8, ptr %next
  br i1 %tst2, label %bb1, label %bb2

bb1:
  store volatile ptr %next, ptr undef
  ret void

bb2:
  call void @bar(ptr %next)
  ret void

end:
  ret void
}

define ptr @test_simple_store_pre(ptr %ptr) {
; CHECK-LABEL: name: test_simple_store_pre
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[VAL:%.*]]:_(s8) = G_CONSTANT i8 0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: [[NEXT:%.*]]:_(p0) = G_INDEXED_STORE [[VAL]](s8), [[BASE]], [[OFFSET]](s64), 1
; CHECK: $x0 = COPY [[NEXT]](p0)

  %next = getelementptr i8, ptr %ptr, i32 42
  store volatile i8 0, ptr %next
  ret ptr %next
}

; The potentially pre-indexed address is used as the value stored. Converting
; would produce the value too late but only by one instruction.
define ptr @test_store_pre_val_loop(ptr %ptr) {
; CHECK-LABEL: name: test_store_pre_val_loop
; CHECK: G_PTR_ADD
; CHECK: G_STORE %

  %next = getelementptr ptr, ptr %ptr, i32 42
  store volatile ptr %next, ptr %next
  ret ptr %next
}

; Potentially pre-indexed address is used between GEP computing it and load.
define ptr @test_load_pre_before(ptr %ptr) {
; CHECK-LABEL: name: test_load_pre_before
; CHECK: G_PTR_ADD
; CHECK: BL @bar
; CHECK: G_LOAD %

  %next = getelementptr i8, ptr %ptr, i32 42
  call void @bar(ptr %next)
  load volatile i8, ptr %next
  ret ptr %next
}

; Materializing the base into a writable register (from sp/fp) would be just as
; bad as the original GEP.
define ptr @test_alloca_load_pre() {
; CHECK-LABEL: name: test_alloca_load_pre
; CHECK: G_PTR_ADD
; CHECK: G_LOAD %

  %ptr = alloca i8, i32 128
  %next = getelementptr i8, ptr %ptr, i32 42
  load volatile i8, ptr %next
  ret ptr %next
}

; Load does not dominate use of its address. No indexing.
define ptr @test_pre_nodom(ptr %in, i1 %tst) {
; CHECK-LABEL: name: test_pre_nodom
; CHECK: G_PTR_ADD
; CHECK: G_LOAD %

  %next = getelementptr i8, ptr %in, i32 16
  br i1 %tst, label %do_indexed, label %use_addr

do_indexed:
  %val = load i8, ptr %next
  store i8 %val, ptr @var
  store ptr %next, ptr @varp8
  br label %use_addr

use_addr:
  ret ptr %next
}

define ptr @test_simple_load_post(ptr %ptr) {
; CHECK-LABEL: name: test_simple_load_post
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: {{%.*}}:_(s8), [[NEXT:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 0
; CHECK: $x0 = COPY [[NEXT]](p0)

  %next = getelementptr i8, ptr %ptr, i32 42
  load volatile i8, ptr %ptr
  ret ptr %next
}

define ptr @test_simple_load_post_gep_after(ptr %ptr) {
; CHECK-LABEL: name: test_simple_load_post_gep_after
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: BL @get_offset
; CHECK: [[OFFSET:%.*]]:_(s64) = COPY $x0
; CHECK: {{%.*}}:_(s8), [[ADDR:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 0
; CHECK: $x0 = COPY [[ADDR]](p0)

  %offset = call i64 @get_offset()
  load volatile i8, ptr %ptr
  %next = getelementptr i8, ptr %ptr, i64 %offset
  ret ptr %next
}

define ptr @test_load_post_keep_looking(ptr %ptr) {
; CHECK: name: test_load_post_keep_looking
; CHECK: G_INDEXED_LOAD

  %offset = call i64 @get_offset()
  load volatile i8, ptr %ptr
  %intval = ptrtoint ptr %ptr to i8
  store i8 %intval, ptr @var

  %next = getelementptr i8, ptr %ptr, i64 %offset
  ret ptr %next
}

; Base is frame index. Using indexing would need copy anyway.
define ptr @test_load_post_alloca() {
; CHECK-LABEL: name: test_load_post_alloca
; CHECK: G_PTR_ADD
; CHECK: G_LOAD %

  %ptr = alloca i8, i32 128
  %next = getelementptr i8, ptr %ptr, i32 42
  load volatile i8, ptr %ptr
  ret ptr %next
}

; Offset computation does not dominate the load we might be indexing.
define ptr @test_load_post_gep_offset_after(ptr %ptr) {
; CHECK-LABEL: name: test_load_post_gep_offset_after
; CHECK: G_LOAD %
; CHECK: BL @get_offset
; CHECK: G_PTR_ADD

  load volatile i8, ptr %ptr
  %offset = call i64 @get_offset()
  %next = getelementptr i8, ptr %ptr, i64 %offset
  ret ptr %next
}

declare void @bar(ptr)
declare i64 @get_offset()
@var = global i8 0
@varp8 = global ptr null
