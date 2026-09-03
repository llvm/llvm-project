; RUN: llc < %s -asm-verbose=false | FileCheck %s

target triple = "wasm32-unknown-unknown"

; --- 0xfc20: memtag.status ---
; CHECK-LABEL: test_status:
; CHECK: memtag.status 0
define i32 @test_status() {
  %1 = call i32 @llvm.wasm.memtag.status.i32(i32 0)
  ret i32 %1
}

; --- 0xfc21: memtag.extract ---
; CHECK-LABEL: test_extract:
; CHECK: local.get 0
; CHECK: memtag.extract 0
define i32 @test_extract(ptr %p) {
  %1 = call i32 @llvm.wasm.memtag.extract.i32(i32 0, ptr %p)
  ret i32 %1
}

; --- 0xfc22: memtag.insert ---
; CHECK-LABEL: test_insert:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.insert 0
define ptr @test_insert(ptr %p, i32 %v) {
  %1 = call ptr @llvm.wasm.memtag.insert.i32(i32 0, ptr %p, i32 %v)
  ret ptr %1
}

; --- 0xfc23: memtag.tagbits ---
; CHECK-LABEL: test_tagbits:
; CHECK: memtag.tagbits 0
define i32 @test_tagbits() {
  %1 = call i32 @llvm.wasm.memtag.tagbits.i32(i32 0)
  ret i32 %1
}

; --- 0xfc24: memtag.startbit ---
; CHECK-LABEL: test_startbit:
; CHECK: memtag.startbit 0
define i32 @test_startbit() {
  %1 = call i32 @llvm.wasm.memtag.startbit.i32(i32 0)
  ret i32 %1
}

; --- 0xfc25: memtag.copy ---
; CHECK-LABEL: test_copy:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.copy 0
define ptr @test_copy(ptr %d, ptr %s) {
  %1 = call ptr @llvm.wasm.memtag.copy(i32 0, ptr %d, ptr %s)
  ret ptr %1
}

; --- 0xfc26: memtag.sub ---
; CHECK-LABEL: test_sub:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.sub 0
define i32 @test_sub(ptr %a, ptr %b) {
  %1 = call i32 @llvm.wasm.memtag.sub.i32(i32 0, ptr %a, ptr %b)
  ret i32 %1
}

; --- 0xfc27: memtag.load ---
; CHECK-LABEL: test_load:
; CHECK: local.get 0
; CHECK: memtag.load 0
define ptr @test_load(ptr %p) {
  %1 = call ptr @llvm.wasm.memtag.load(i32 0, ptr %p)
  ret ptr %1
}

; --- 0xfc28: memtag.untag ---
; CHECK-LABEL: test_untag:
; CHECK: local.get 0
; CHECK: memtag.untag 0
define ptr @test_untag(ptr %p) {
  %1 = call ptr @llvm.wasm.memtag.untag(i32 0, ptr %p)
  ret ptr %1
}

; --- 0xfc29: memtag.untagstore ---
; CHECK-LABEL: test_untagstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.untagstore 0
define ptr @test_untagstore(ptr %p, i32 %v) {
  %1 = call ptr @llvm.wasm.memtag.untagstore.i32(i32 0, ptr %p, i32 %v)
  ret ptr %1
}

; --- 0xfc2a: memtag.untagstorez ---
; CHECK-LABEL: test_untagstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.untagstorez 0
define ptr @test_untagstorez(ptr %p, i32 %v) {
  %1 = call ptr @llvm.wasm.memtag.untagstorez.i32(i32 0, ptr %p, i32 %v)
  ret ptr %1
}

; --- 0xfc2b: memtag.store ---
; CHECK-LABEL: test_store:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.store 0
define void @test_store(ptr %p, i32 %v) {
  call void @llvm.wasm.memtag.store.i32(i32 0, ptr %p, i32 %v)
  ret void
}

; --- 0xfc2c: memtag.storez ---
; CHECK-LABEL: test_storez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.storez 0
define void @test_storez(ptr %p, i32 %v) {
  call void @llvm.wasm.memtag.storez.i32(i32 0, ptr %p, i32 %v)
  ret void
}

; --- 0xfc2d: memtag.random ---
; CHECK-LABEL: test_random:
; CHECK: local.get 0
; CHECK: memtag.random 0
define ptr @test_random(ptr %p) {
  %1 = call ptr @llvm.wasm.memtag.random(i32 0, ptr %p)
  ret ptr %1
}

; --- 0xfc2e: memtag.randomstore ---
; CHECK-LABEL: test_randomstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.randomstore 0
define ptr @test_randomstore(ptr %p, i32 %v) {
  %1 = call ptr @llvm.wasm.memtag.randomstore.i32(i32 0, ptr %p, i32 %v)
  ret ptr %1
}

; --- 0xfc2f: memtag.randomstorez ---
; CHECK-LABEL: test_randomstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.randomstorez 0
define ptr @test_randomstorez(ptr %p, i32 %v) {
  %1 = call ptr @llvm.wasm.memtag.randomstorez.i32(i32 0, ptr %p, i32 %v)
  ret ptr %1
}

; --- 0xfc30: memtag.randommask ---
; CHECK-LABEL: test_randommask:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: memtag.randommask 0
define ptr @test_randommask(ptr %p, i32 %m) {
  %1 = call ptr @llvm.wasm.memtag.randommask.i32(i32 0, ptr %p, i32 %m)
  ret ptr %1
}

; --- 0xfc31: memtag.randommaskstore ---
; CHECK-LABEL: test_randommaskstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memtag.randommaskstore 0
define ptr @test_randommaskstore(ptr %p, i32 %v, i32 %m) {
  %1 = call ptr @llvm.wasm.memtag.randommaskstore.i32(i32 0, ptr %p, i32 %v, i32 %m)
  ret ptr %1
}

; --- 0xfc32: memtag.randommaskstorez ---
; CHECK-LABEL: test_randommaskstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memtag.randommaskstorez 0
define ptr @test_randommaskstorez(ptr %p, i32 %v, i32 %m) {
  %1 = call ptr @llvm.wasm.memtag.randommaskstorez.i32(i32 0, ptr %p, i32 %v, i32 %m)
  ret ptr %1
}

; --- 0xfc33: memtag.add ---
; CHECK-LABEL: test_add:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memtag.add 0
define ptr @test_add(ptr %p, i32 %o, i32 %t) {
  %1 = call ptr @llvm.wasm.memtag.add.i32(i32 0, ptr %p, i32 %o, i32 %t)
  ret ptr %1
}

; --- 0xfc34: memtag.addstore ---
; CHECK-LABEL: test_addstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memtag.addstore 0
define ptr @test_addstore(ptr %p, i32 %v, i32 %o, i32 %t) {
  %1 = call ptr @llvm.wasm.memtag.addstore.i32(i32 0, ptr %p, i32 %v, i32 %o, i32 %t)
  ret ptr %1
}

; --- 0xfc35: memtag.addstorez ---
; CHECK-LABEL: test_addstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memtag.addstorez 0
define ptr @test_addstorez(ptr %p, i32 %v, i32 %o, i32 %t) {
  %1 = call ptr @llvm.wasm.memtag.addstorez.i32(i32 0, ptr %p, i32 %v, i32 %o, i32 %t)
  ret ptr %1
}

; --- 0xfc36: memtag.hint ---
; CHECK-LABEL: test_hint:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: memtag.hint 0
define ptr @test_hint(ptr %p, ptr %h, i32 %i) {
  %1 = call ptr @llvm.wasm.memtag.hint.i32(i32 0, ptr %p, ptr %h, i32 %i)
  ret ptr %1
}

; --- 0xfc37: memtag.hintstore ---
; CHECK-LABEL: test_hintstore:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memtag.hintstore 0
define ptr @test_hintstore(ptr %p, i32 %v, ptr %h, i32 %i) {
  %1 = call ptr @llvm.wasm.memtag.hintstore.i32(i32 0, ptr %p, i32 %v, ptr %h, i32 %i)
  ret ptr %1
}

; --- 0xfc38: memtag.hintstorez ---
; CHECK-LABEL: test_hintstorez:
; CHECK: local.get 0
; CHECK: local.get 1
; CHECK: local.get 2
; CHECK: local.get 3
; CHECK: memtag.hintstorez 0
define ptr @test_hintstorez(ptr %p, i32 %v, ptr %h, i32 %i) {
  %1 = call ptr @llvm.wasm.memtag.hintstorez.i32(i32 0, ptr %p, i32 %v, ptr %h, i32 %i)
  ret ptr %1
}

; --- Intrinsic Declarations ---
declare i32 @llvm.wasm.memtag.status.i32(i32)
declare i32 @llvm.wasm.memtag.extract.i32(i32, ptr)
declare ptr @llvm.wasm.memtag.insert.i32(i32, ptr, i32)
declare i32 @llvm.wasm.memtag.tagbits.i32(i32)
declare i32 @llvm.wasm.memtag.startbit.i32(i32)
declare ptr @llvm.wasm.memtag.copy(i32, ptr, ptr)
declare i32 @llvm.wasm.memtag.sub.i32(i32, ptr, ptr)
declare ptr @llvm.wasm.memtag.load(i32, ptr)
declare ptr @llvm.wasm.memtag.untag(i32, ptr)
declare ptr @llvm.wasm.memtag.untagstore.i32(i32, ptr, i32)
declare ptr @llvm.wasm.memtag.untagstorez.i32(i32, ptr, i32)
declare void @llvm.wasm.memtag.store.i32(i32, ptr, i32)
declare void @llvm.wasm.memtag.storez.i32(i32, ptr, i32)
declare ptr @llvm.wasm.memtag.random(i32, ptr)
declare ptr @llvm.wasm.memtag.randomstore.i32(i32, ptr, i32)
declare ptr @llvm.wasm.memtag.randomstorez.i32(i32, ptr, i32)
declare ptr @llvm.wasm.memtag.randommask.i32(i32, ptr, i32)
declare ptr @llvm.wasm.memtag.randommaskstore.i32(i32, ptr, i32, i32)
declare ptr @llvm.wasm.memtag.randommaskstorez.i32(i32, ptr, i32, i32)
declare ptr @llvm.wasm.memtag.add.i32(i32, ptr, i32, i32)
declare ptr @llvm.wasm.memtag.addstore.i32(i32, ptr, i32, i32, i32)
declare ptr @llvm.wasm.memtag.addstorez.i32(i32, ptr, i32, i32, i32)
declare ptr @llvm.wasm.memtag.hint.i32(i32, ptr, ptr, i32)
declare ptr @llvm.wasm.memtag.hintstore.i32(i32, ptr, i32, ptr, i32)
declare ptr @llvm.wasm.memtag.hintstorez.i32(i32, ptr, i32, ptr, i32)
