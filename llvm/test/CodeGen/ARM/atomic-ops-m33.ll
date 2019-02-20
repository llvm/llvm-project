; RUN: llc -mtriple=thumbv7-none-eabi -mcpu=cortex-m33 -verify-machineinstrs -o -  %s | FileCheck %s

define i8 @test_atomic_load_add_i8(i8 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i8:
  %old = atomicrmw add i8* @var8, i8 %offset seq_cst
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var8
; CHECK: movt r[[ADDR]], :upper16:var8

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexb r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlexb [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
  ret i8 %old
}

define i16 @test_atomic_load_add_i16(i16 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i16:
  %old = atomicrmw add i16* @var16, i16 %offset acquire
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var16
; CHECK: movt r[[ADDR]], :upper16:var16

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldaexh r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: strexh [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
  ret i16 %old
}

define i32 @test_atomic_load_add_i32(i32 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i32:
  %old = atomicrmw add i32* @var32, i32 %offset release
; CHECK-NOT: dmb
; CHECK-NOT: mcr
; CHECK: movw r[[ADDR:[0-9]+]], :lower16:var32
; CHECK: movt r[[ADDR]], :upper16:var32

; CHECK: .LBB{{[0-9]+}}_1:
; CHECK: ldrex r[[OLD:[0-9]+]], [r[[ADDR]]]
  ; r0 below is a reasonable guess but could change: it certainly comes into the
  ;  function there.
; CHECK-NEXT: add{{s?}} [[NEW:r[0-9]+]], r[[OLD]], r0
; CHECK-NEXT: stlex [[STATUS:r[0-9]+]], [[NEW]], [r[[ADDR]]]
; CHECK-NEXT: cmp [[STATUS]], #0
; CHECK-NEXT: bne .LBB{{[0-9]+}}_1
; CHECK-NOT: dmb
; CHECK-NOT: mcr

; CHECK: mov r0, r[[OLD]]
  ret i32 %old
}

define void @test_atomic_load_add_i64(i64 %offset) nounwind {
; CHECK-LABEL: test_atomic_load_add_i64:
; CHECK: bl __sync_fetch_and_add_8
   %old = atomicrmw add i64* @var64, i64 %offset monotonic
  store i64 %old, i64* @var64
  ret void
}

define i8 @test_load_acquire_i8(i8* %ptr) {
; CHECK-LABEL: test_load_acquire_i8:
; CHECK: ldab r0, [r0]
  %val = load atomic i8, i8* %ptr seq_cst, align 1
  ret i8 %val
}

define i16 @test_load_acquire_i16(i16* %ptr) {
; CHECK-LABEL: test_load_acquire_i16:
; CHECK: ldah r0, [r0]
  %val = load atomic i16, i16* %ptr acquire, align 2
  ret i16 %val
}

define i32 @test_load_acquire_i32(i32* %ptr) {
; CHECK-LABEL: test_load_acquire_i32:
; CHECK: lda r0, [r0]
  %val = load atomic i32, i32* %ptr acquire, align 4
  ret i32 %val
}

define i64 @test_load_acquire_i64(i64* %ptr) {
; CHECK-LABEL: test_load_acquire_i64:
; CHECK: bl __atomic_load
  %val = load atomic i64, i64* %ptr acquire, align 4
  ret i64 %val
}

define void @test_store_release_i8(i8 %val, i8* %ptr) {
; CHECK-LABEL: test_store_release_i8:
; CHECK: stlb r0, [r1]
  store atomic i8 %val, i8* %ptr seq_cst, align 1
  ret void
}

define void @test_store_release_i16(i16 %val, i16* %ptr) {
; CHECK-LABEL: test_store_release_i16:
; CHECK: stlh r0, [r1]
  store atomic i16 %val, i16* %ptr release, align 2
  ret void
}

define void @test_store_release_i32(i32 %val, i32* %ptr) {
; CHECK-LABEL: test_store_release_i32:
; CHECK: stl r0, [r1]
  store atomic i32 %val, i32* %ptr seq_cst, align 4
  ret void
}

define void @test_store_release_i64(i64 %val, i64* %ptr) {
; CHECK-LABEL: test_store_release_i64:
; CHECK: bl __atomic_store
  store atomic i64 %val, i64* %ptr seq_cst, align 4
  ret void
}


@var8 = global i8 0
@var16 = global i16 0
@var32 = global i32 0
@var64 = global i64 0
