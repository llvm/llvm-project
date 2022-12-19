; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+v8.4a %s -o - | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+lse2 %s -o - | FileCheck %s

define void @test_atomic_load(ptr %addr) {
; CHECK-LABEL: test_atomic_load:

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: stp [[LO]], [[HI]], [x0]
  %res.0 = load atomic i128, ptr %addr monotonic, align 16
  store i128 %res.0, ptr %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: stp [[LO]], [[HI]], [x0]
  %res.1 = load atomic i128, ptr %addr unordered, align 16
  store i128 %res.1, ptr %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: dmb ish
; CHECK: stp [[LO]], [[HI]], [x0]
  %res.2 = load atomic i128, ptr %addr acquire, align 16
  store i128 %res.2, ptr %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: dmb ish
; CHECK: stp [[LO]], [[HI]], [x0]
  %res.3 = load atomic i128, ptr %addr seq_cst, align 16
  store i128 %res.3, ptr %addr



; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0, #32]
; CHECK-DAG: stp [[LO]], [[HI]], [x0]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 32
  %res.5 = load atomic i128, ptr %addr8.1 monotonic, align 16
  store i128 %res.5, ptr %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0, #504]
; CHECK: stp [[LO]], [[HI]], [x0]
  %addr8.2 = getelementptr i8,  ptr %addr, i32 504
  %res.6 = load atomic i128, ptr %addr8.2 monotonic, align 16
  store i128 %res.6, ptr %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0, #-512]
; CHECK: stp [[LO]], [[HI]], [x0]
  %addr8.3 = getelementptr i8,  ptr %addr, i32 -512
  %res.7 = load atomic i128, ptr %addr8.3 monotonic, align 16
  store i128 %res.7, ptr %addr

  ret void
}

define void @test_libcall_load(ptr %addr) {
; CHECK-LABEL: test_libcall_load:
; CHECK: bl __atomic_load
  %res.8 = load atomic i128, ptr %addr unordered, align 8
  store i128 %res.8, ptr %addr

  ret void
}

define void @test_nonfolded_load1(ptr %addr) {
; CHECK-LABEL: test_nonfolded_load1:

; CHECK: add x[[ADDR:[0-9]+]], x0, #4
; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x[[ADDR]]]
; CHECK: stp [[LO]], [[HI]], [x0]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 4
  %res.1 = load atomic i128, ptr %addr8.1 monotonic, align 16
  store i128 %res.1, ptr %addr

  ret void
}

define void @test_nonfolded_load2(ptr %addr) {
; CHECK-LABEL: test_nonfolded_load2:

; CHECK: add x[[ADDR:[0-9]+]], x0, #512
; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x[[ADDR]]]
; CHECK: stp [[LO]], [[HI]], [x0]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 512
  %res.1 = load atomic i128, ptr %addr8.1 monotonic, align 16
  store i128 %res.1, ptr %addr

  ret void
}

define void @test_nonfolded_load3(ptr %addr) {
; CHECK-LABEL: test_nonfolded_load3:

; CHECK: sub x[[ADDR:[0-9]+]], x0, #520
; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x[[ADDR]]]
; CHECK: stp [[LO]], [[HI]], [x0]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 -520
  %res.1 = load atomic i128, ptr %addr8.1 monotonic, align 16
  store i128 %res.1, ptr %addr

  ret void
}

define void @test_atomic_store(ptr %addr, i128 %val) {
; CHECK-LABEL: test_atomic_store:

; CHECK: stp x2, x3, [x0]
  store atomic i128 %val, ptr %addr monotonic, align 16

; CHECK: stp x2, x3, [x0]
  store atomic i128 %val, ptr %addr unordered, align 16

; CHECK: dmb ish
; CHECK: stp x2, x3, [x0]
  store atomic i128 %val, ptr %addr release, align 16

; CHECK: dmb ish
; CHECK: stp x2, x3, [x0]
; CHECK: dmb ish
  store atomic i128 %val, ptr %addr seq_cst, align 16



; CHECK: stp x2, x3, [x0, #8]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 8
  store atomic i128 %val, ptr %addr8.1 monotonic, align 16

; CHECK: stp x2, x3, [x0, #504]
  %addr8.2 = getelementptr i8,  ptr %addr, i32 504
  store atomic i128 %val, ptr %addr8.2 monotonic, align 16

; CHECK: stp x2, x3, [x0, #-512]
  %addr8.3 = getelementptr i8,  ptr %addr, i32 -512
  store atomic i128 %val, ptr %addr8.3 monotonic, align 16

  ret void
}

define void @test_libcall_store(ptr %addr, i128 %val) {
; CHECK-LABEL: test_libcall_store:
; CHECK: bl __atomic_store
  store atomic i128 %val, ptr %addr unordered, align 8

  ret void
}

define void @test_nonfolded_store1(ptr %addr, i128 %val) {
; CHECK-LABEL: test_nonfolded_store1:

; CHECK: add x[[ADDR:[0-9]+]], x0, #4
; CHECK: stp x2, x3, [x[[ADDR]]]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 4
  store atomic i128 %val, ptr %addr8.1 monotonic, align 16

  ret void
}

define void @test_nonfolded_store2(ptr %addr, i128 %val) {
; CHECK-LABEL: test_nonfolded_store2:

; CHECK: add x[[ADDR:[0-9]+]], x0, #512
; CHECK: stp x2, x3, [x[[ADDR]]]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 512
  store atomic i128 %val, ptr %addr8.1 monotonic, align 16

  ret void
}

define void @test_nonfolded_store3(ptr %addr, i128 %val) {
; CHECK-LABEL: test_nonfolded_store3:

; CHECK: sub x[[ADDR:[0-9]+]], x0, #520
; CHECK: stp x2, x3, [x[[ADDR]]]
  %addr8.1 = getelementptr i8,  ptr %addr, i32 -520
  store atomic i128 %val, ptr %addr8.1 monotonic, align 16

  ret void
}
