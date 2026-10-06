; RUN: opt < %s -S -mtriple=powerpc--linux -passes=instrprof -instrprof-atomic-counter-update-all | FileCheck %s --check-prefix=CHECK,BE
; RUN: opt < %s -S -mtriple=powerpcle--linux -passes=instrprof -instrprof-atomic-counter-update-all | FileCheck %s --check-prefix=CHECK,LE

target triple = "powerpc--linux"

@__profn_foo = private constant [3 x i8] c"foo"

; CHECK-LABEL: define void @foo

; BE-NEXT: %[[LOW:[0-9]+]] = atomicrmw add ptr getelementptr inbounds (i32, ptr @__profc_foo, i32 1), i32 1 monotonic, align 4
; LE-NEXT: %[[LOW:[0-9]+]] = atomicrmw add ptr @__profc_foo, i32 1 monotonic, align 4
; CHECK-NEXT: %[[LOW_UPD:[0-9]+]] = add i32 %[[LOW]], 1
; CHECK-NEXT: %[[CMP:[0-9]+]] = icmp ult i32 %[[LOW_UPD]], %[[LOW]]
; CHECK-NEXT: %[[ZEXT:[0-9]+]] = zext i1 %[[CMP]] to i32
; CHECK-NEXT: %[[INC:[0-9]+]] = add i32 0, %[[ZEXT]]
; CHECK-NEXT: %pgocount.ifnonzero = icmp ne i32 %[[INC]], 0
; CHECK-NEXT: br i1 %pgocount.ifnonzero, label %[[BR1:[0-9]+]], label %[[BR2:[0-9]+]]

; CHECK: [[BR1:[0-9]+]]:
; BE-NEXT: %[[HIGH:[0-9]+]] = atomicrmw add ptr @__profc_foo, i32 %[[INC]] monotonic, align 4
; LE-NEXT: %[[HIGH:[0-9]+]] = atomicrmw add ptr getelementptr inbounds (i32, ptr @__profc_foo, i32 1), i32 %[[INC]] monotonic, align 4
; CHECK-NEXT: br label %[[BR2:[0-9]+]]

; CHECK: [[BR2]]:
; CHECK-NEXT: ret void

define void @foo() {
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 0, i32 1, i32 0)
  ret void
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
