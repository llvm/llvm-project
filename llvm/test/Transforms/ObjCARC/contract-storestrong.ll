; RUN: opt -passes=objc-arc-contract -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

declare ptr @llvm.objc.retain(ptr)
declare void @llvm.objc.release(ptr)
declare void @use_pointer(ptr)

@x = external global ptr

; CHECK-LABEL: define void @test0(
; CHECK: entry:
; CHECK-NEXT: tail call void @llvm.objc.storeStrong(ptr @x, ptr %p) [[NUW:#[0-9]+]]
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test0(ptr %p) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %p) nounwind
  %tmp = load ptr, ptr @x, align 8
  store ptr %0, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %tmp) nounwind
  ret void
}

; Don't do this if the load is volatile.

; CHECK-LABEL: define void @test1(ptr %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call ptr @llvm.objc.retain(ptr %p) [[NUW]]
; CHECK-NEXT:   %tmp = load volatile ptr, ptr @x, align 8
; CHECK-NEXT:   store ptr %0, ptr @x, align 8
; CHECK-NEXT:   tail call void @llvm.objc.release(ptr %tmp) [[NUW]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test1(ptr %p) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %p) nounwind
  %tmp = load volatile ptr, ptr @x, align 8
  store ptr %0, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %tmp) nounwind
  ret void
}

; Don't do this if the store is volatile.

; CHECK-LABEL: define void @test2(ptr %p) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call ptr @llvm.objc.retain(ptr %p) [[NUW]]
; CHECK-NEXT:   %tmp = load ptr, ptr @x, align 8
; CHECK-NEXT:   store volatile ptr %0, ptr @x, align 8
; CHECK-NEXT:   tail call void @llvm.objc.release(ptr %tmp) [[NUW]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test2(ptr %p) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %p) nounwind
  %tmp = load ptr, ptr @x, align 8
  store volatile ptr %0, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %tmp) nounwind
  ret void
}

; Don't do this if there's a use of the old pointer value between the store
; and the release.

; CHECK-LABEL: define void @test3(ptr %newValue) {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) [[NUW]]
; CHECK-NEXT:    %x1 = load ptr, ptr @x, align 8
; CHECK-NEXT:    store ptr %x0, ptr @x, align 8
; CHECK-NEXT:    tail call void @use_pointer(ptr %x1), !clang.arc.no_objc_arc_exceptions !0
; CHECK-NEXT:    tail call void @llvm.objc.release(ptr %x1) [[NUW]], !clang.imprecise_release !0
; CHECK-NEXT:    ret void
; CHECK-NEXT:  }
define void @test3(ptr %newValue) {
entry:
  %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) nounwind
  %x1 = load ptr, ptr @x, align 8
  store ptr %newValue, ptr @x, align 8
  tail call void @use_pointer(ptr %x1), !clang.arc.no_objc_arc_exceptions !0
  tail call void @llvm.objc.release(ptr %x1) nounwind, !clang.imprecise_release !0
  ret void
}

; Like test3, but with an icmp use instead of a call, for good measure.

; CHECK-LABEL:  define i1 @test4(ptr %newValue, ptr %foo) {
; CHECK-NEXT:   entry:
; CHECK-NEXT:     %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) [[NUW]]
; CHECK-NEXT:     %x1 = load ptr, ptr @x, align 8
; CHECK-NEXT:     store ptr %x0, ptr @x, align 8
; CHECK-NEXT:     %t = icmp eq ptr %x1, %foo
; CHECK-NEXT:     tail call void @llvm.objc.release(ptr %x1) [[NUW]], !clang.imprecise_release !0
; CHECK-NEXT:     ret i1 %t
; CHECK-NEXT:   }
define i1 @test4(ptr %newValue, ptr %foo) {
entry:
  %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) nounwind
  %x1 = load ptr, ptr @x, align 8
  store ptr %newValue, ptr @x, align 8
  %t = icmp eq ptr %x1, %foo
  tail call void @llvm.objc.release(ptr %x1) nounwind, !clang.imprecise_release !0
  ret i1 %t
}

; Do form an llvm.objc.storeStrong here, because the use is before the store.

; CHECK-LABEL: define i1 @test5(ptr %newValue, ptr %foo) {
; CHECK: %t = icmp eq ptr %x1, %foo
; CHECK: tail call void @llvm.objc.storeStrong(ptr @x, ptr %newValue) [[NUW]]
; CHECK: }
define i1 @test5(ptr %newValue, ptr %foo) {
entry:
  %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) nounwind
  %x1 = load ptr, ptr @x, align 8
  %t = icmp eq ptr %x1, %foo
  store ptr %newValue, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %x1) nounwind, !clang.imprecise_release !0
  ret i1 %t
}

; Like test5, but the release is before the store.

; CHECK-LABEL: define i1 @test6(ptr %newValue, ptr %foo) {
; CHECK: %t = icmp eq ptr %x1, %foo
; CHECK: tail call void @llvm.objc.storeStrong(ptr @x, ptr %newValue) [[NUW]]
; CHECK: }
define i1 @test6(ptr %newValue, ptr %foo) {
entry:
  %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) nounwind
  %x1 = load ptr, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %x1) nounwind, !clang.imprecise_release !0
  %t = icmp eq ptr %x1, %foo
  store ptr %newValue, ptr @x, align 8
  ret i1 %t
}

; Like test0, but there's no store, so don't form an llvm.objc.storeStrong.

; CHECK-LABEL: define void @test7(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %0 = tail call ptr @llvm.objc.retain(ptr %p) [[NUW]]
; CHECK-NEXT:   %tmp = load ptr, ptr @x, align 8
; CHECK-NEXT:   tail call void @llvm.objc.release(ptr %tmp) [[NUW]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test7(ptr %p) {
entry:
  %0 = tail call ptr @llvm.objc.retain(ptr %p) nounwind
  %tmp = load ptr, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %tmp) nounwind
  ret void
}

; Like test0, but there's no retain, so don't form an llvm.objc.storeStrong.

; CHECK-LABEL: define void @test8(
; CHECK-NEXT: entry:
; CHECK-NEXT:   %tmp = load ptr, ptr @x, align 8
; CHECK-NEXT:   store ptr %p, ptr @x, align 8
; CHECK-NEXT:   tail call void @llvm.objc.release(ptr %tmp) [[NUW]]
; CHECK-NEXT:   ret void
; CHECK-NEXT: }
define void @test8(ptr %p) {
entry:
  %tmp = load ptr, ptr @x, align 8
  store ptr %p, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %tmp) nounwind
  ret void
}

; Make sure that we properly handle release that *may* release our new
; value in between the retain and the store. We need to be sure that
; this we can safely move the retain to the store. This specific test
; makes sure that we properly handled a release of an unrelated
; pointer.
;
; CHECK-LABEL: define i1 @test9(ptr %newValue, ptr %foo, ptr %unrelated_ptr) {
; CHECK-NOT: llvm.objc.storeStrong
define i1 @test9(ptr %newValue, ptr %foo, ptr %unrelated_ptr) {
entry:
  %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) nounwind
  tail call void @llvm.objc.release(ptr %unrelated_ptr) nounwind, !clang.imprecise_release !0
  %x1 = load ptr, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %x1) nounwind, !clang.imprecise_release !0
  %t = icmp eq ptr %x1, %foo
  store ptr %newValue, ptr @x, align 8
  ret i1 %t
}

; Make sure that we don't perform the optimization when we just have a call.
;
; CHECK-LABEL: define i1 @test10(ptr %newValue, ptr %foo, ptr %unrelated_ptr) {
; CHECK-NOT: llvm.objc.storeStrong
define i1 @test10(ptr %newValue, ptr %foo, ptr %unrelated_ptr) {
entry:
  %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) nounwind
  call void @use_pointer(ptr %unrelated_ptr)
  %x1 = load ptr, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %x1) nounwind, !clang.imprecise_release !0
  %t = icmp eq ptr %x1, %foo
  store ptr %newValue, ptr @x, align 8
  ret i1 %t
}

; Make sure we form the store strong if the use in between the retain
; and the store does not touch reference counts.
; CHECK-LABEL: define i1 @test11(ptr %newValue, ptr %foo, ptr %unrelated_ptr) {
; CHECK: llvm.objc.storeStrong
define i1 @test11(ptr %newValue, ptr %foo, ptr %unrelated_ptr) {
entry:
  %x0 = tail call ptr @llvm.objc.retain(ptr %newValue) nounwind
  %t = icmp eq ptr %newValue, %foo
  %x1 = load ptr, ptr @x, align 8
  tail call void @llvm.objc.release(ptr %x1) nounwind, !clang.imprecise_release !0
  store ptr %newValue, ptr @x, align 8
  ret i1 %t
}

; Make sure that we form the store strong even if there are bitcasts on
; the pointers.
; CHECK-LABEL: define void @test12(
; CHECK: entry:
; CHECK-NEXT: %p16 = bitcast ptr @x to ptr
; CHECK-NEXT: %tmp16 = load ptr, ptr %p16, align 8
; CHECK-NEXT: %tmp8 = bitcast ptr %tmp16 to ptr
; CHECK-NEXT: %p32 = bitcast ptr @x to ptr
; CHECK-NEXT: %v32 = bitcast ptr %p to ptr
; CHECK-NEXT: tail call void @llvm.objc.storeStrong(ptr %p16, ptr %p)
; CHECK-NEXT: ret void
; CHECK-NEXT: }
define void @test12(ptr %p) {
entry:
  %retain = tail call ptr @llvm.objc.retain(ptr %p) nounwind
  %p16 = bitcast ptr @x to ptr
  %tmp16 = load ptr, ptr %p16, align 8
  %tmp8 = bitcast ptr %tmp16 to ptr
  %p32 = bitcast ptr @x to ptr
  %v32 = bitcast ptr %retain to ptr
  store ptr %v32, ptr %p32, align 8
  tail call void @llvm.objc.release(ptr %tmp8) nounwind
  ret void
}

; This used to crash.
; CHECK-LABEL: define ptr @test13(
; CHECK: tail call void @llvm.objc.storeStrong(ptr %{{.*}}, ptr %[[NEW:.*]])
; CHECK-NEXT: ret ptr %[[NEW]]

define ptr @test13(ptr %a0, ptr %a1, ptr %addr, ptr %new) {
  %old = load ptr, ptr %addr, align 8
  call void @llvm.objc.release(ptr %old)
  %retained = call ptr @llvm.objc.retain(ptr %new)
  store ptr %retained, ptr %addr, align 8
  ret ptr %retained
}

; Cannot form a storeStrong call because it's unsafe to move the release call to
; the store.

; CHECK-LABEL: define void @test14(
; CHECK: %[[V0:.*]] = load ptr, ptr %a
; CHECK: %[[V1:.*]] = call ptr @llvm.objc.retain(ptr %p)
; CHECK: store ptr %[[V1]], ptr %a
; CHECK: %[[V2:.*]] = call ptr @llvm.objc.retain(ptr %[[V0]])
; CHECK: call void @llvm.objc.release(ptr %[[V2]])

define void @test14(ptr %a, ptr %p) {
  %v0 = load ptr, ptr %a, align 8
  %v1 = call ptr @llvm.objc.retain(ptr %p)
  store ptr %p, ptr %a, align 8
  %v2  = call ptr @llvm.objc.retain(ptr %v0)
  call void @llvm.objc.release(ptr %v0)
  ret void
}

!0 = !{}

; CHECK: attributes [[NUW]] = { nounwind }
