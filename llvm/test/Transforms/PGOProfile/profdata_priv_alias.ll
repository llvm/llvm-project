; RUN: opt -S -passes=pgo-instr-gen,instrprof < %s | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Test that we use private aliases to reference function addresses inside profile data
; CHECK: @__profd_foo = private global {{.*}} ptr @foo.local
; CHECK-NOT: @__profd_foo = private global {{.*}} ptr @foo

; CHECK: @__profd_weak = private global {{.*}} ptr @weak.local
; CHECK: @__profd_linkonce = private global {{.*}} ptr @linkonce.local
; CHECK: @__profd_weakodr = private global {{.*}} ptr @weakodr.local
; CHECK: @__profd_linkonceodr = private global {{.*}} ptr @linkonceodr.local

; available_externally shouldn't have an alias, so make sure it doesn't appear here
; CHECK: @__profc_available_externally.[[HASH:[#0-9]+]]
; CHECK-NOT: @__profd_available_externally.[[HASH]] = {{.*}}ptr @available_externally.[[HASH]].local

;; Ensure when not instrumenting a non-comdat function, then if we generate an
;; alias, then it is private. We check comdat versions in comdat.ll
; CHECK: @foo.local = private alias i32 (i32), ptr @foo
; CHECK: @weak.local = private alias void (), ptr @weak
; CHECK: @linkonce.local = private alias void (), ptr @linkonce
; CHECK: @weakodr.local = private alias void (), ptr @weakodr
; CHECK: @linkonceodr.local = private alias void (), ptr @linkonceodr

;; We should never generate an alias for available_externally functions
; CHECK-NOT: @available_externally{{.*}} = private alias void (), ptr @available_externally

define i32 @foo(i32 %0) {
; CHECK-LABEL: @foo(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_foo, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[PGOCOUNT]], 1
; CHECK-NEXT:    store i64 [[TMP1]], ptr @__profc_foo, align 8
; CHECK-NEXT:    ret i32 0
entry:
  ret i32 0
}

define weak void @weak() {
; CHECK-LABEL: @weak(
; CHECK-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_weak, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[PGOCOUNT]], 1
; CHECK-NEXT:    store i64 [[TMP1]], ptr @__profc_weak, align 8
; CHECK-NEXT:    ret void
  ret void
}

define linkonce void @linkonce() {
; CHECK-LABEL: @linkonce(
; CHECK-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_linkonce, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[PGOCOUNT]], 1
; CHECK-NEXT:    store i64 [[TMP1]], ptr @__profc_linkonce, align 8
; CHECK-NEXT:    ret void
  ret void
}

define weak_odr void @weakodr() {
; CHECK-LABEL: @weakodr(
; CHECK-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_weakodr, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[PGOCOUNT]], 1
; CHECK-NEXT:    store i64 [[TMP1]], ptr @__profc_weakodr, align 8
; CHECK-NEXT:    ret void
  ret void
}

define linkonce_odr void @linkonceodr() {
; CHECK-LABEL: @linkonceodr(
; CHECK-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_linkonceodr, align 8
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[PGOCOUNT]], 1
; CHECK-NEXT:    store i64 [[TMP1]], ptr @__profc_linkonceodr, align 8
; CHECK-NEXT:    ret void
  ret void
}

define available_externally void @available_externally(){
; CHECK-LABEL: @available_externally(
; CHECK-NEXT:    [[PGOCOUNT:%.*]] = load i64, ptr @__profc_available_externally.[[HASH]], align 8
; CHECK-NEXT:    [[TMP1:%.*]] = add i64 [[PGOCOUNT]], 1
; CHECK-NEXT:    store i64 [[TMP1]], ptr @__profc_available_externally.[[HASH]], align 8
; CHECK-NEXT:    ret void
  ret void
}
