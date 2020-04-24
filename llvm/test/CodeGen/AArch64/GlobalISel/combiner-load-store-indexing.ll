; RUN: llc -mtriple=arm64-apple-ios -global-isel -global-isel-abort=1 -verify-machineinstrs -stop-after=aarch64-prelegalizer-combiner -force-legal-indexing %s -o - | FileCheck %s

define i8* @test_simple_load_pre(i8* %ptr) {
; CHECK-LABEL: name: test_simple_load_pre
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: {{%.*}}:_(s8), [[NEXT:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 1
; CHECK: $x0 = COPY [[NEXT]](p0)

  %next = getelementptr i8, i8* %ptr, i32 42
  load volatile i8, i8* %next
  ret i8* %next
}

define void @test_load_multiple_dominated(i8* %ptr, i1 %tst, i1 %tst2) {
; CHECK-LABEL: name: test_load_multiple_dominated
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: {{%.*}}:_(s8), [[NEXT:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 1
; CHECK: $x0 = COPY [[NEXT]](p0)
  %next = getelementptr i8, i8* %ptr, i32 42
  br i1 %tst, label %do_load, label %end

do_load:
  load volatile i8, i8* %next
  br i1 %tst2, label %bb1, label %bb2

bb1:
  store volatile i8* %next, i8** undef
  ret void

bb2:
  call void @bar(i8* %next)
  ret void

end:
  ret void
}

define i8* @test_simple_store_pre(i8* %ptr) {
; CHECK-LABEL: name: test_simple_store_pre
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[VAL:%.*]]:_(s8) = G_CONSTANT i8 0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: [[NEXT:%.*]]:_(p0) = G_INDEXED_STORE [[VAL]](s8), [[BASE]], [[OFFSET]](s64), 1
; CHECK: $x0 = COPY [[NEXT]](p0)

  %next = getelementptr i8, i8* %ptr, i32 42
  store volatile i8 0, i8* %next
  ret i8* %next
}

; The potentially pre-indexed address is used as the value stored. Converting
; would produce the value too late but only by one instruction.
define i64** @test_store_pre_val_loop(i64** %ptr) {
; CHECK-LABEL: name: test_store_pre_val_loop
; CHECK: G_PTR_ADD
; CHECK: G_STORE %

  %next = getelementptr i64*, i64** %ptr, i32 42
  %next.p0 = bitcast i64** %next to i64*
  store volatile i64* %next.p0, i64** %next
  ret i64** %next
}

; Potentially pre-indexed address is used between GEP computing it and load.
define i8* @test_load_pre_before(i8* %ptr) {
; CHECK-LABEL: name: test_load_pre_before
; CHECK: G_PTR_ADD
; CHECK: BL @bar
; CHECK: G_LOAD %

  %next = getelementptr i8, i8* %ptr, i32 42
  call void @bar(i8* %next)
  load volatile i8, i8* %next
  ret i8* %next
}

; Materializing the base into a writable register (from sp/fp) would be just as
; bad as the original GEP.
define i8* @test_alloca_load_pre() {
; CHECK-LABEL: name: test_alloca_load_pre
; CHECK: G_PTR_ADD
; CHECK: G_LOAD %

  %ptr = alloca i8, i32 128
  %next = getelementptr i8, i8* %ptr, i32 42
  load volatile i8, i8* %next
  ret i8* %next
}

; Load does not dominate use of its address. No indexing.
define i8* @test_pre_nodom(i8* %in, i1 %tst) {
; CHECK-LABEL: name: test_pre_nodom
; CHECK: G_PTR_ADD
; CHECK: G_LOAD %

  %next = getelementptr i8, i8* %in, i32 16
  br i1 %tst, label %do_indexed, label %use_addr

do_indexed:
  %val = load i8, i8* %next
  store i8 %val, i8* @var
  store i8* %next, i8** @varp8
  br label %use_addr

use_addr:
  ret i8* %next
}

define i8* @test_simple_load_post(i8* %ptr) {
; CHECK-LABEL: name: test_simple_load_post
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: [[OFFSET:%.*]]:_(s64) = G_CONSTANT i64 42
; CHECK-NOT: G_PTR_ADD
; CHECK: {{%.*}}:_(s8), [[NEXT:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 0
; CHECK: $x0 = COPY [[NEXT]](p0)

  %next = getelementptr i8, i8* %ptr, i32 42
  load volatile i8, i8* %ptr
  ret i8* %next
}

define i8* @test_simple_load_post_gep_after(i8* %ptr) {
; CHECK-LABEL: name: test_simple_load_post_gep_after
; CHECK: [[BASE:%.*]]:_(p0) = COPY $x0
; CHECK: BL @get_offset
; CHECK: [[OFFSET:%.*]]:_(s64) = COPY $x0
; CHECK: {{%.*}}:_(s8), [[ADDR:%.*]]:_(p0) = G_INDEXED_LOAD [[BASE]], [[OFFSET]](s64), 0
; CHECK: $x0 = COPY [[ADDR]](p0)

  %offset = call i64 @get_offset()
  load volatile i8, i8* %ptr
  %next = getelementptr i8, i8* %ptr, i64 %offset
  ret i8* %next
}

define i8* @test_load_post_keep_looking(i8* %ptr) {
; CHECK: name: test_load_post_keep_looking
; CHECK: G_INDEXED_LOAD

  %offset = call i64 @get_offset()
  load volatile i8, i8* %ptr
  %intval = ptrtoint i8* %ptr to i8
  store i8 %intval, i8* @var

  %next = getelementptr i8, i8* %ptr, i64 %offset
  ret i8* %next
}

; Base is frame index. Using indexing would need copy anyway.
define i8* @test_load_post_alloca() {
; CHECK-LABEL: name: test_load_post_alloca
; CHECK: G_PTR_ADD
; CHECK: G_LOAD %

  %ptr = alloca i8, i32 128
  %next = getelementptr i8, i8* %ptr, i32 42
  load volatile i8, i8* %ptr
  ret i8* %next
}

; Offset computation does not dominate the load we might be indexing.
define i8* @test_load_post_gep_offset_after(i8* %ptr) {
; CHECK-LABEL: name: test_load_post_gep_offset_after
; CHECK: G_LOAD %
; CHECK: BL @get_offset
; CHECK: G_PTR_ADD

  load volatile i8, i8* %ptr
  %offset = call i64 @get_offset()
  %next = getelementptr i8, i8* %ptr, i64 %offset
  ret i8* %next
}

declare void @bar(i8*)
declare i64 @get_offset()
@var = global i8 0
@varp8 = global i8* null
