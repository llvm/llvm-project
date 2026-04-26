; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test all WebAssembly Memory Tagging intrinsics on wasm64

target triple = "wasm64-unknown-unknown"

; --- Status & Configuration ---

; CHECK-LABEL: test_status:
; CHECK: local.get 0
; CHECK: memory.memtag_status
define i64 @test_status(i32 %idx) {
  %1 = call i64 @llvm.wasm.memtag.status.i64(i32 %idx)
  ret i64 %1
}

; CHECK-LABEL: test_tagbits:
; CHECK: local.get 0
; CHECK: memory.memtag_tagbits
define i64 @test_tagbits(i32 %idx) {
  %1 = call i64 @llvm.wasm.memtag.tagbits.i64(i32 %idx)
  ret i64 %1
}

; CHECK-LABEL: test_startbit:
; CHECK: local.get 0
; CHECK: memory.memtag_startbit
define i64 @test_startbit(i32 %idx) {
  %1 = call i64 @llvm.wasm.memtag.startbit.i64(i32 %idx)
  ret i64 %1
}

; --- Pointer Manipulation ---

; CHECK-LABEL: test_extract:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memory.memtag_extract
define i64 @test_extract(i32 %idx, ptr %p) {
  %1 = call i64 @llvm.wasm.memtag.extract.i64(i32 %idx, ptr %p)
  ret i64 %1
}

; CHECK-LABEL: test_insert:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_insert
define ptr @test_insert(i32 %idx, ptr %p, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.insert.i64(i32 %idx, ptr %p, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_untag:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memory.memtag_untag
define ptr @test_untag(i32 %idx, ptr %p) {
  %1 = call ptr @llvm.wasm.memtag.untag(i32 %idx, ptr %p)
  ret ptr %1
}

; CHECK-LABEL: test_add:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memory.memtag_add
define ptr @test_add(i32 %idx, ptr %p, i64 %offs, i64 %mask) {
  %1 = call ptr @llvm.wasm.memtag.add.i64(i32 %idx, ptr %p, i64 %offs, i64 %mask)
  ret ptr %1
}

; CHECK-LABEL: test_sub:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_sub
define i64 @test_sub(i32 %idx, ptr %p0, ptr %p1) {
  %1 = call i64 @llvm.wasm.memtag.sub.i64(i32 %idx, ptr %p0, ptr %p1)
  ret i64 %1
}

; --- Load / Store / Copy ---

; CHECK-LABEL: test_load:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memory.memtag_load
define ptr @test_load(i32 %idx, ptr %p) {
  %1 = call ptr @llvm.wasm.memtag.load(i32 %idx, ptr %p)
  ret ptr %1
}

; CHECK-LABEL: test_store:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_store
define void @test_store(i32 %idx, ptr %p, i64 %tag) {
  call void @llvm.wasm.memtag.store.i64(i32 %idx, ptr %p, i64 %tag)
  ret void
}

; CHECK-LABEL: test_storez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_storez
define void @test_storez(i32 %idx, ptr %p, i64 %tag) {
  call void @llvm.wasm.memtag.storez.i64(i32 %idx, ptr %p, i64 %tag)
  ret void
}

; CHECK-LABEL: test_untagstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_untagstore
define ptr @test_untagstore(i32 %idx, ptr %p, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.untagstore.i64(i32 %idx, ptr %p, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_untagstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_untagstorez
define ptr @test_untagstorez(i32 %idx, ptr %p, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.untagstorez.i64(i32 %idx, ptr %p, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_copy:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_copy
define ptr @test_copy(i32 %idx, ptr %dest, ptr %src) {
  %1 = call ptr @llvm.wasm.memtag.copy(i32 %idx, ptr %dest, ptr %src)
  ret ptr %1
}

; --- Randomization ---

; CHECK-LABEL: test_random:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memory.memtag_random
define ptr @test_random(i32 %idx, ptr %p) {
  %1 = call ptr @llvm.wasm.memtag.random(i32 %idx, ptr %p)
  ret ptr %1
}

; CHECK-LABEL: test_randomstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_randomstore
define ptr @test_randomstore(i32 %idx, ptr %p, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.randomstore.i64(i32 %idx, ptr %p, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_randomstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_randomstorez
define ptr @test_randomstorez(i32 %idx, ptr %p, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.randomstorez.i64(i32 %idx, ptr %p, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_randommask:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memory.memtag_randommask
define ptr @test_randommask(i32 %idx, ptr %p, i64 %mask) {
  %1 = call ptr @llvm.wasm.memtag.randommask.i64(i32 %idx, ptr %p, i64 %mask)
  ret ptr %1
}

; CHECK-LABEL: test_randommaskstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memory.memtag_randommaskstore
define ptr @test_randommaskstore(i32 %idx, ptr %p, i64 %mask, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.randommaskstore.i64(i32 %idx, ptr %p, i64 %mask, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_randommaskstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memory.memtag_randommaskstorez
define ptr @test_randommaskstorez(i32 %idx, ptr %p, i64 %mask, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.randommaskstorez.i64(i32 %idx, ptr %p, i64 %mask, i64 %tag)
  ret ptr %1
}

; --- Compound Operations & Hints ---

; CHECK-LABEL: test_addstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: local.get 4
; CHECK: memory.memtag_addstore
define ptr @test_addstore(i32 %idx, ptr %p, i64 %offs, i64 %mask, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.addstore.i64(i32 %idx, ptr %p, i64 %offs, i64 %mask, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_addstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: local.get 4
; CHECK: memory.memtag_addstorez
define ptr @test_addstorez(i32 %idx, ptr %p, i64 %offs, i64 %mask, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.addstorez.i64(i32 %idx, ptr %p, i64 %offs, i64 %mask, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_hint:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memory.memtag_hint
define ptr @test_hint(i32 %idx, ptr %p0, ptr %p1, i64 %tag) {
  %1 = call ptr @llvm.wasm.memtag.hint.i64(i32 %idx, ptr %p0, ptr %p1, i64 %tag)
  ret ptr %1
}

; CHECK-LABEL: test_hintstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: local.get 4
; CHECK: memory.memtag_hintstore
define ptr @test_hintstore(i32 %idx, ptr %p0, i64 %tag0, ptr %p1, i64 %tag1) {
  %1 = call ptr @llvm.wasm.memtag.hintstore.i64(i32 %idx, ptr %p0, i64 %tag0, ptr %p1, i64 %tag1)
  ret ptr %1
}

; CHECK-LABEL: test_hintstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: local.get 4
; CHECK: memory.memtag_hintstorez
define ptr @test_hintstorez(i32 %idx, ptr %p0, i64 %tag0, ptr %p1, i64 %tag1) {
  %1 = call ptr @llvm.wasm.memtag.hintstorez.i64(i32 %idx, ptr %p0, i64 %tag0, ptr %p1, i64 %tag1)
  ret ptr %1
}

; Declarations
declare i64 @llvm.wasm.memtag.status.i64(i32)
declare i64 @llvm.wasm.memtag.tagbits.i64(i32)
declare i64 @llvm.wasm.memtag.startbit.i64(i32)
declare i64 @llvm.wasm.memtag.extract.i64(i32, ptr)
declare ptr @llvm.wasm.memtag.insert.i64(i32, ptr, i64)
declare ptr @llvm.wasm.memtag.untag(i32, ptr)
declare ptr @llvm.wasm.memtag.add.i64(i32, ptr, i64, i64)
declare i64 @llvm.wasm.memtag.sub.i64(i32, ptr, ptr)
declare ptr @llvm.wasm.memtag.load(i32, ptr)
declare void @llvm.wasm.memtag.store.i64(i32, ptr, i64)
declare void @llvm.wasm.memtag.storez.i64(i32, ptr, i64)
declare ptr @llvm.wasm.memtag.untagstore.i64(i32, ptr, i64)
declare ptr @llvm.wasm.memtag.untagstorez.i64(i32, ptr, i64)
declare ptr @llvm.wasm.memtag.copy(i32, ptr, ptr)
declare ptr @llvm.wasm.memtag.random(i32, ptr)
declare ptr @llvm.wasm.memtag.randomstore.i64(i32, ptr, i64)
declare ptr @llvm.wasm.memtag.randomstorez.i64(i32, ptr, i64)
declare ptr @llvm.wasm.memtag.randommask.i64(i32, ptr, i64)
declare ptr @llvm.wasm.memtag.randommaskstore.i64(i32, ptr, i64, i64)
declare ptr @llvm.wasm.memtag.randommaskstorez.i64(i32, ptr, i64, i64)
declare ptr @llvm.wasm.memtag.addstore.i64(i32, ptr, i64, i64, i64)
declare ptr @llvm.wasm.memtag.addstorez.i64(i32, ptr, i64, i64, i64)
declare ptr @llvm.wasm.memtag.hint.i64(i32, ptr, ptr, i64)
declare ptr @llvm.wasm.memtag.hintstore.i64(i32, ptr, i64, ptr, i64)
declare ptr @llvm.wasm.memtag.hintstorez.i64(i32, ptr, i64, ptr, i64)
