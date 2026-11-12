; RUN: opt -passes=objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare ptr @llvm.objc.retain(ptr)
declare ptr @llvm.objc.retainAutorelease(ptr)
declare void @llvm.objc.release(ptr)
declare ptr @llvm.objc.autorelease(ptr)

declare void @llvm.objc.clang.arc.use(...)
declare void @llvm.objc.clang.arc.noop.use(...)

declare void @test0_helper(ptr, ptr)
declare void @can_release(ptr)

; Ensure that we honor clang.arc.use as a use and don't miscompile
; the reduced test case from <rdar://13195034>.
;
; CHECK-LABEL:      define void @test0(
; CHECK:        @llvm.objc.retain(ptr %x)
; CHECK-NEXT:   store ptr %y, ptr %temp0
; CHECK-NEXT:   @llvm.objc.retain(ptr %y)
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL1:%.*]] = load ptr, ptr %temp0
; CHECK-NEXT:   @llvm.objc.retain(ptr [[VAL1]])
; CHECK-NEXT:   call void (...) @llvm.objc.clang.arc.use(ptr %y)
; CHECK-NEXT:   @llvm.objc.release(ptr %y)
; CHECK-NEXT:   store ptr [[VAL1]], ptr %temp1
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL2:%.*]] = load ptr, ptr %temp1
; CHECK-NEXT:   @llvm.objc.retain(ptr [[VAL2]])
; CHECK-NEXT:   call void (...) @llvm.objc.clang.arc.use(ptr [[VAL1]])
; CHECK-NEXT:   @llvm.objc.release(ptr [[VAL1]])
; CHECK-NEXT:   @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:   store ptr %x, ptr %out
; CHECK-NEXT:   @llvm.objc.retain(ptr %x)
; CHECK-NEXT:   @llvm.objc.release(ptr [[VAL2]])
; CHECK-NEXT:   @llvm.objc.release(ptr %x)
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test0(ptr %out, ptr %x, ptr %y) {
entry:
  %temp0 = alloca ptr, align 8
  %temp1 = alloca ptr, align 8
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %1 = call ptr @llvm.objc.retain(ptr %y) nounwind
  store ptr %y, ptr %temp0
  call void @test0_helper(ptr %x, ptr %temp0)
  %val1 = load ptr, ptr %temp0
  %2 = call ptr @llvm.objc.retain(ptr %val1) nounwind
  call void (...) @llvm.objc.clang.arc.use(ptr %y) nounwind
  call void @llvm.objc.release(ptr %y) nounwind
  store ptr %val1, ptr %temp1
  call void @test0_helper(ptr %x, ptr %temp1)
  %val2 = load ptr, ptr %temp1
  %3 = call ptr @llvm.objc.retain(ptr %val2) nounwind
  call void (...) @llvm.objc.clang.arc.use(ptr %val1) nounwind
  call void @llvm.objc.release(ptr %val1) nounwind
  %4 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %5 = call ptr @llvm.objc.autorelease(ptr %x) nounwind
  store ptr %x, ptr %out
  call void @llvm.objc.release(ptr %val2) nounwind
  call void @llvm.objc.release(ptr %x) nounwind
  ret void
}

; CHECK-LABEL:      define void @test0a(
; CHECK:        @llvm.objc.retain(ptr %x)
; CHECK-NEXT:   store ptr %y, ptr %temp0
; CHECK-NEXT:   @llvm.objc.retain(ptr %y)
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL1:%.*]] = load ptr, ptr %temp0
; CHECK-NEXT:   @llvm.objc.retain(ptr [[VAL1]])
; CHECK-NEXT:   call void (...) @llvm.objc.clang.arc.use(ptr %y)
; CHECK-NEXT:   @llvm.objc.release(ptr %y)
; CHECK-NEXT:   store ptr [[VAL1]], ptr %temp1
; CHECK-NEXT:   call void @test0_helper
; CHECK-NEXT:   [[VAL2:%.*]] = load ptr, ptr %temp1
; CHECK-NEXT:   @llvm.objc.retain(ptr [[VAL2]])
; CHECK-NEXT:   call void (...) @llvm.objc.clang.arc.use(ptr [[VAL1]])
; CHECK-NEXT:   @llvm.objc.release(ptr [[VAL1]])
; CHECK-NEXT:   @llvm.objc.autorelease(ptr %x)
; CHECK-NEXT:   @llvm.objc.release(ptr [[VAL2]])
; CHECK-NEXT:   store ptr %x, ptr %out
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test0a(ptr %out, ptr %x, ptr %y) {
entry:
  %temp0 = alloca ptr, align 8
  %temp1 = alloca ptr, align 8
  %0 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %1 = call ptr @llvm.objc.retain(ptr %y) nounwind
  store ptr %y, ptr %temp0
  call void @test0_helper(ptr %x, ptr %temp0)
  %val1 = load ptr, ptr %temp0
  %2 = call ptr @llvm.objc.retain(ptr %val1) nounwind
  call void (...) @llvm.objc.clang.arc.use(ptr %y) nounwind
  call void @llvm.objc.release(ptr %y) nounwind, !clang.imprecise_release !0
  store ptr %val1, ptr %temp1
  call void @test0_helper(ptr %x, ptr %temp1)
  %val2 = load ptr, ptr %temp1
  %3 = call ptr @llvm.objc.retain(ptr %val2) nounwind
  call void (...) @llvm.objc.clang.arc.use(ptr %val1) nounwind
  call void @llvm.objc.release(ptr %val1) nounwind, !clang.imprecise_release !0
  %4 = call ptr @llvm.objc.retain(ptr %x) nounwind
  %5 = call ptr @llvm.objc.autorelease(ptr %x) nounwind
  store ptr %x, ptr %out
  call void @llvm.objc.release(ptr %val2) nounwind, !clang.imprecise_release !0
  call void @llvm.objc.release(ptr %x) nounwind, !clang.imprecise_release !0
  ret void
}

; ARC optimizer should be able to safely remove the retain/release pair as the
; call to @llvm.objc.clang.arc.noop.use is a no-op.

; CHECK-LABEL: define void @test_arc_noop_use(
; CHECK-NEXT:    call void @can_release(ptr %x)
; CHECK-NEXT:    call void (...) @llvm.objc.clang.arc.noop.use(
; CHECK-NEXT:    ret void

define void @test_arc_noop_use(ptr %out, ptr %x) {
  call ptr @llvm.objc.retain(ptr %x)
  call void @can_release(ptr %x)
  call void (...) @llvm.objc.clang.arc.noop.use(ptr %x)
  call void @llvm.objc.release(ptr %x), !clang.imprecise_release !0
  ret void
}

!0 = !{}

