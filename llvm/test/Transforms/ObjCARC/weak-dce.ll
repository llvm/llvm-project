; RUN: opt -S -basicaa -objc-arc < %s | FileCheck %s
; rdar://11434915

; Delete the weak calls and replace them with just the net retain.

;      CHECK: define void @test0(i8* %p) {
; CHECK-NEXT: call i8* @objc_retain(i8* %p)
; CHECK-NEXT: ret void

define void @test0(i8* %p) {
  %weakBlock = alloca i8*, align 8
  %tmp7 = call i8* @objc_initWeak(i8** %weakBlock, i8* %p) nounwind
  %tmp26 = call i8* @objc_loadWeakRetained(i8** %weakBlock) nounwind
  call void @objc_destroyWeak(i8** %weakBlock) nounwind
  ret void
}

;      CHECK: define i8* @test1(i8* %p) {
; CHECK-NEXT: call i8* @objc_retain(i8* %p)
; CHECK-NEXT: ret i8* %p

define i8* @test1(i8* %p) {
  %weakBlock = alloca i8*, align 8
  %tmp7 = call i8* @objc_initWeak(i8** %weakBlock, i8* %p) nounwind
  %tmp26 = call i8* @objc_loadWeakRetained(i8** %weakBlock) nounwind
  call void @objc_destroyWeak(i8** %weakBlock) nounwind
  ret i8* %tmp26
}

;      CHECK: define i8* @test2(i8* %p, i8* %q) {
; CHECK-NEXT: call i8* @objc_retain(i8* %q)
; CHECK-NEXT: ret i8* %q

define i8* @test2(i8* %p, i8* %q) {
  %weakBlock = alloca i8*, align 8
  %tmp7 = call i8* @objc_initWeak(i8** %weakBlock, i8* %p) nounwind
  %tmp19 = call i8* @objc_storeWeak(i8** %weakBlock, i8* %q) nounwind
  %tmp26 = call i8* @objc_loadWeakRetained(i8** %weakBlock) nounwind
  call void @objc_destroyWeak(i8** %weakBlock) nounwind
  ret i8* %tmp26
}

declare i8* @objc_initWeak(i8**, i8*)
declare void @objc_destroyWeak(i8**)
declare i8* @objc_loadWeakRetained(i8**)
declare i8* @objc_storeWeak(i8** %weakBlock, i8* %q)
