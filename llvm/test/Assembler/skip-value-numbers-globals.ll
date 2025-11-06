; RUN: opt -S < %s | FileCheck %s

@5 = global i8 0
@"" = global i8 1
@10 = alias i8, ptr @5
@15 = ifunc ptr(), ptr @20

define ptr @20() {
  ret ptr null
}

declare void @25()

declare ptr @""(ptr)

define void @test(ptr %p) {
  store ptr @5, ptr %p
  store ptr @6, ptr %p
  store ptr @10, ptr %p
  store ptr @15, ptr %p
  store ptr @20, ptr %p
  store ptr @25, ptr %p
  store ptr @26, ptr %p
  ret void
}

; CHECK: @0 = global i8 0
; CHECK: @1 = global i8 1
; CHECK: @2 = alias i8, ptr @0
; CHECK: @3 = ifunc ptr (), ptr @4

; CHECK-LABEL: define ptr @4() {
; CHECK-NEXT:  ret ptr null

; CHECK: declare void @5()

; CHECK: declare ptr @6(ptr)

; CHECK-LABEL: define void @test(ptr %p) {
; CHECK-NEXT: store ptr @0, ptr %p, align 8
; CHECK-NEXT: store ptr @1, ptr %p, align 8
; CHECK-NEXT: store ptr @2, ptr %p, align 8
; CHECK-NEXT: store ptr @3, ptr %p, align 8
; CHECK-NEXT: store ptr @4, ptr %p, align 8
; CHECK-NEXT: store ptr @5, ptr %p, align 8
; CHECK-NEXT: store ptr @6, ptr %p, align 8
; CHECK-NEXT: ret void
