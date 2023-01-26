; RUN: opt -S -passes=objc-arc < %s | FileCheck %s
; rdar://11434915

; Delete the weak calls and replace them with just the net retain.

;      CHECK: define void @test0(ptr %p) {
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: ret void

define void @test0(ptr %p) {
  %weakBlock = alloca ptr, align 8
  %tmp7 = call ptr @llvm.objc.initWeak(ptr %weakBlock, ptr %p) nounwind
  %tmp26 = call ptr @llvm.objc.loadWeakRetained(ptr %weakBlock) nounwind
  call void @llvm.objc.destroyWeak(ptr %weakBlock) nounwind
  ret void
}

;      CHECK: define ptr @test1(ptr %p) {
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %p)
; CHECK-NEXT: ret ptr %p

define ptr @test1(ptr %p) {
  %weakBlock = alloca ptr, align 8
  %tmp7 = call ptr @llvm.objc.initWeak(ptr %weakBlock, ptr %p) nounwind
  %tmp26 = call ptr @llvm.objc.loadWeakRetained(ptr %weakBlock) nounwind
  call void @llvm.objc.destroyWeak(ptr %weakBlock) nounwind
  ret ptr %tmp26
}

;      CHECK: define ptr @test2(ptr %p, ptr %q) {
; CHECK-NEXT: call ptr @llvm.objc.retain(ptr %q)
; CHECK-NEXT: ret ptr %q

define ptr @test2(ptr %p, ptr %q) {
  %weakBlock = alloca ptr, align 8
  %tmp7 = call ptr @llvm.objc.initWeak(ptr %weakBlock, ptr %p) nounwind
  %tmp19 = call ptr @llvm.objc.storeWeak(ptr %weakBlock, ptr %q) nounwind
  %tmp26 = call ptr @llvm.objc.loadWeakRetained(ptr %weakBlock) nounwind
  call void @llvm.objc.destroyWeak(ptr %weakBlock) nounwind
  ret ptr %tmp26
}

declare ptr @llvm.objc.initWeak(ptr, ptr)
declare void @llvm.objc.destroyWeak(ptr)
declare ptr @llvm.objc.loadWeakRetained(ptr)
declare ptr @llvm.objc.storeWeak(ptr %weakBlock, ptr %q)
