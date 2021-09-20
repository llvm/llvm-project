; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+v8.4a %s -o - -global-isel=1 -global-isel-abort=1 | FileCheck %s
; RUN: llc -mtriple=aarch64-linux-gnu -mattr=+lse2 %s -o - -global-isel=1 -global-isel-abort=1 | FileCheck %s

define void @test_atomic_load(i128* %addr) {
; CHECK-LABEL: test_atomic_load:

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %res.0 = load atomic i128, i128* %addr monotonic, align 16
  store i128 %res.0, i128* %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %res.1 = load atomic i128, i128* %addr unordered, align 16
  store i128 %res.1, i128* %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: dmb ish
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %res.2 = load atomic i128, i128* %addr acquire, align 16
  store i128 %res.2, i128* %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0]
; CHECK: dmb ish
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %res.3 = load atomic i128, i128* %addr seq_cst, align 16
  store i128 %res.3, i128* %addr

  %addr8 = bitcast i128* %addr to i8*

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0, #8]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 8
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  %res.5 = load atomic i128, i128* %addr128.1 monotonic, align 16
  store i128 %res.5, i128* %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0, #504]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %addr8.2 = getelementptr i8,  i8* %addr8, i32 504
  %addr128.2 = bitcast i8* %addr8.2 to i128*
  %res.6 = load atomic i128, i128* %addr128.2 monotonic, align 16
  store i128 %res.6, i128* %addr

; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x0, #-512]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %addr8.3 = getelementptr i8,  i8* %addr8, i32 -512
  %addr128.3 = bitcast i8* %addr8.3 to i128*
  %res.7 = load atomic i128, i128* %addr128.3 monotonic, align 16
  store i128 %res.7, i128* %addr

  ret void
}

define void @test_libcall_load(i128* %addr) {
; CHECK-LABEL: test_libcall_load:
; CHECK: bl __atomic_load
  %res.8 = load atomic i128, i128* %addr unordered, align 8
  store i128 %res.8, i128* %addr

  ret void
}

define void @test_nonfolded_load1(i128* %addr) {
; CHECK-LABEL: test_nonfolded_load1:
  %addr8 = bitcast i128* %addr to i8*

; CHECK: add x[[ADDR:[0-9]+]], x0, #4
; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x[[ADDR]]]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 4
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  %res.1 = load atomic i128, i128* %addr128.1 monotonic, align 16
  store i128 %res.1, i128* %addr

  ret void
}

define void @test_nonfolded_load2(i128* %addr) {
; CHECK-LABEL: test_nonfolded_load2:
  %addr8 = bitcast i128* %addr to i8*

; CHECK: add x[[ADDR:[0-9]+]], x0, #512
; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x[[ADDR]]]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 512
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  %res.1 = load atomic i128, i128* %addr128.1 monotonic, align 16
  store i128 %res.1, i128* %addr

  ret void
}

define void @test_nonfolded_load3(i128* %addr) {
; CHECK-LABEL: test_nonfolded_load3:
  %addr8 = bitcast i128* %addr to i8*

; CHECK: sub x[[ADDR:[0-9]+]], x0, #520
; CHECK: ldp [[LO:x[0-9]+]], [[HI:x[0-9]+]], [x[[ADDR]]]
; CHECK: mov v[[Q:[0-9]+]].d[0], [[LO]]
; CHECK: mov v[[Q]].d[1], [[HI]]
; CHECK: str q[[Q]], [x0]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 -520
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  %res.1 = load atomic i128, i128* %addr128.1 monotonic, align 16
  store i128 %res.1, i128* %addr

  ret void
}

define void @test_atomic_store(i128* %addr, i128 %val) {
; CHECK-LABEL: test_atomic_store:

; CHECK: stp x2, x3, [x0]
  store atomic i128 %val, i128* %addr monotonic, align 16

; CHECK: stp x2, x3, [x0]
  store atomic i128 %val, i128* %addr unordered, align 16

; CHECK: dmb ish
; CHECK: stp x2, x3, [x0]
  store atomic i128 %val, i128* %addr release, align 16

; CHECK: dmb ish
; CHECK: stp x2, x3, [x0]
; CHECK: dmb ish
  store atomic i128 %val, i128* %addr seq_cst, align 16

  %addr8 = bitcast i128* %addr to i8*

; CHECK: stp x2, x3, [x0, #8]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 8
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  store atomic i128 %val, i128* %addr128.1 monotonic, align 16

; CHECK: stp x2, x3, [x0, #504]
  %addr8.2 = getelementptr i8,  i8* %addr8, i32 504
  %addr128.2 = bitcast i8* %addr8.2 to i128*
  store atomic i128 %val, i128* %addr128.2 monotonic, align 16

; CHECK: stp x2, x3, [x0, #-512]
  %addr8.3 = getelementptr i8,  i8* %addr8, i32 -512
  %addr128.3 = bitcast i8* %addr8.3 to i128*
  store atomic i128 %val, i128* %addr128.3 monotonic, align 16

  ret void
}

define void @test_libcall_store(i128* %addr, i128 %val) {
; CHECK-LABEL: test_libcall_store:
; CHECK: bl __atomic_store
  store atomic i128 %val, i128* %addr unordered, align 8

  ret void
}

define void @test_nonfolded_store1(i128* %addr, i128 %val) {
; CHECK-LABEL: test_nonfolded_store1:
  %addr8 = bitcast i128* %addr to i8*

; CHECK: add x[[ADDR:[0-9]+]], x0, #4
; CHECK: stp x2, x3, [x[[ADDR]]]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 4
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  store atomic i128 %val, i128* %addr128.1 monotonic, align 16

  ret void
}

define void @test_nonfolded_store2(i128* %addr, i128 %val) {
; CHECK-LABEL: test_nonfolded_store2:
  %addr8 = bitcast i128* %addr to i8*

; CHECK: add x[[ADDR:[0-9]+]], x0, #512
; CHECK: stp x2, x3, [x[[ADDR]]]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 512
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  store atomic i128 %val, i128* %addr128.1 monotonic, align 16

  ret void
}

define void @test_nonfolded_store3(i128* %addr, i128 %val) {
; CHECK-LABEL: test_nonfolded_store3:
  %addr8 = bitcast i128* %addr to i8*

; CHECK: sub x[[ADDR:[0-9]+]], x0, #520
; CHECK: stp x2, x3, [x[[ADDR]]]
  %addr8.1 = getelementptr i8,  i8* %addr8, i32 -520
  %addr128.1 = bitcast i8* %addr8.1 to i128*
  store atomic i128 %val, i128* %addr128.1 monotonic, align 16

  ret void
}
