; RUN: llc < %s -verify-machineinstrs -global-isel=0 \
; RUN:   -mtriple arm64e-apple-darwin | FileCheck %s --check-prefixes=CHECK,DARWIN
; RUN: llc < %s -verify-machineinstrs -global-isel=1 -global-isel-abort=1 \
; RUN:   -mtriple arm64e-apple-darwin | FileCheck %s --check-prefixes=CHECK,DARWIN
; RUN: llc < %s -verify-machineinstrs -global-isel=0 \
; RUN:   -mtriple aarch64 -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,ELF
; RUN: llc < %s -verify-machineinstrs -global-isel=1 -global-isel-abort=1 \
; RUN:   -mtriple aarch64 -mattr=+pauth | FileCheck %s --check-prefixes=CHECK,ELF

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"

; Basic: no discriminator, no offset.

define i64 @test_da(ptr %ptr) {
; CHECK-LABEL: test_da:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldraa x0, [x0]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  ret i64 %tmp3
}

define i64 @test_db(ptr %ptr) {
; CHECK-LABEL: test_db:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldrab x0, [x0]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  ret i64 %tmp3
}

define i64 @test_ia(ptr %ptr) {
; CHECK-LABEL: test_ia:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autiza x16
; CHECK-NEXT:    ldr x0, [x16]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 0, i64 0)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  ret i64 %tmp3
}

define i64 @test_ib(ptr %ptr) {
; CHECK-LABEL: test_ib:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autizb x16
; CHECK-NEXT:    ldr x0, [x16]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 1, i64 0)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  ret i64 %tmp3
}

; No discriminator, interesting offsets.

define i64 @test_da_8(ptr %ptr) {
; CHECK-LABEL: test_da_8:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldraa x0, [x0, #8]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 8
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_db_simm9max(ptr %ptr) {
; CHECK-LABEL: test_db_simm9max:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldrab x0, [x0, #4088]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = add i64 %tmp1, 4088
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_db_uimm12max(ptr %ptr) {
; CHECK-LABEL: test_db_uimm12max:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdzb x16
; CHECK-NEXT:    ldr x0, [x16, #32760]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = add i64 %tmp1, 32760
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_da_4(ptr %ptr) {
; CHECK-LABEL: test_da_4:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdza x16
; CHECK-NEXT:    ldur x0, [x16, #4]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 4
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_da_largeoff_12b(ptr %ptr) {
; CHECK-LABEL: test_da_largeoff_12b:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdza x16
; CHECK-NEXT:    mov x17, #32768
; CHECK-NEXT:    ldr x0, [x16, x17]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 32768
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_da_largeoff_32b(ptr %ptr) {
; CHECK-LABEL: test_da_largeoff_32b:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdza x16
; CHECK-NEXT:    mov x17, #2
; CHECK-NEXT:    movk x17, #1, lsl #32
; CHECK-NEXT:    ldr x0, [x16, x17]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 4294967298
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_da_m8(ptr %ptr) {
; CHECK-LABEL: test_da_m8:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldraa x0, [x0, #-8]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, -8
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_db_simm9min(ptr %ptr) {
; CHECK-LABEL: test_db_simm9min:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldrab x0, [x0, #-4096]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = add i64 %tmp1, -4096
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_da_neg_largeoff_12b(ptr %ptr) {
; CHECK-LABEL: test_da_neg_largeoff_12b:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdza x16
; CHECK-NEXT:    mov x17, #-32768
; CHECK-NEXT:    ldr x0, [x16, x17]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, -32768
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_da_neg_largeoff_32b(ptr %ptr) {
; CHECK-LABEL: test_da_neg_largeoff_32b:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdza x16
; CHECK-NEXT:    mov x17, #-3
; CHECK-NEXT:    movk x17, #65534, lsl #32
; CHECK-NEXT:    ldr x0, [x16, x17]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, -4294967299
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

define i64 @test_da_disc_m256(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_disc_m256:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autda x16, x1
; CHECK-NEXT:    ldur x0, [x16, #-256]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 %disc)
  %tmp2 = add i64 %tmp1, -256
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

; No discriminator, interesting offsets, writeback.

define ptr @test_da_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_da_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldraa x8, [x0, #0]!
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  store i64 %tmp3, ptr %dst
  ret ptr %tmp2
}

define ptr @test_da_8_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_da_8_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldraa x8, [x0, #8]!
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 8
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

define ptr @test_da_simm9max_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_da_simm9max_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldraa x8, [x0, #4088]!
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 4088
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

define ptr @test_da_uimm12max_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_da_uimm12max_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdza x16
; CHECK-NEXT:    mov x17, #32760
; CHECK-NEXT:    add x16, x16, x17
; CHECK-NEXT:    ldr x8, [x16]
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 32760
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

define ptr @test_db_4_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_db_4_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdzb x16
; CHECK-NEXT:    ldr x8, [x16, #4]!
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = add i64 %tmp1, 4
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

define ptr @test_da_largeoff_12b_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_da_largeoff_12b_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdza x16
; CHECK-NEXT:    mov x17, #32768
; CHECK-NEXT:    add x16, x16, x17
; CHECK-NEXT:    ldr x8, [x16]
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 32768
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

define ptr @test_db_m256_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_db_m256_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldrab x8, [x0, #-256]!
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = add i64 %tmp1, -256
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

define ptr @test_db_simm9min_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_db_simm9min_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    ldrab x8, [x0, #-4096]!
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = add i64 %tmp1, -4096
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

define ptr @test_db_neg_largeoff_12b_wb(ptr %ptr, ptr %dst) {
; CHECK-LABEL: test_db_neg_largeoff_12b_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autdzb x16
; CHECK-NEXT:    mov x17, #-32768
; CHECK-NEXT:    add x16, x16, x17
; CHECK-NEXT:    ldr x8, [x16]
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x1]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 3, i64 0)
  %tmp2 = add i64 %tmp1, -32768
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  ret ptr %tmp3
}

; Writeback, with a potential cycle.

define void @test_da_wb_cycle(ptr %ptr, ptr %dst, ptr %dst2, ptr %dst3) {
; DARWIN-LABEL: test_da_wb_cycle:
; DARWIN:       %bb.0:
; DARWIN-NEXT:    mov x16, x0
; DARWIN-NEXT:    autdza x16
; DARWIN-NEXT:    str x16, [x2]
; DARWIN-NEXT:    ldr x8, [x16]
; DARWIN-NEXT:    str x8, [x1]
; DARWIN-NEXT:    ret

; ELF-LABEL: test_da_wb_cycle:
; ELF:       %bb.0:
; ELF-NEXT:    autdza x0
; ELF-NEXT:    str x0, [x2]
; ELF-NEXT:    ldr x8, [x0]
; ELF-NEXT:    str x8, [x1]
; ELF-NEXT:    ret

  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  store i64 %tmp1, ptr %dst2
  %tmp3 = load i64, ptr %tmp2
  store i64 %tmp3, ptr %dst
  ret void
}

; Writeback multiple-use of the auth.

define ptr @test_da_8_wb_use(ptr %ptr, ptr %dst, ptr %dst2) {
; DARWIN-LABEL: test_da_8_wb_use:
; DARWIN:       %bb.0:
; DARWIN-NEXT:    mov x16, x0
; DARWIN-NEXT:    autdza x16
; DARWIN-NEXT:    ldraa x8, [x0, #8]!
; DARWIN-NEXT:    str x8, [x1]
; DARWIN-NEXT:    str x16, [x2]
; DARWIN-NEXT:    ret

; ELF-LABEL: test_da_8_wb_use:
; ELF:       %bb.0:
; ELF-NEXT:    mov x8, x0
; ELF-NEXT:    autdza x8
; ELF-NEXT:    ldraa x9, [x0, #8]!
; ELF-NEXT:    str x9, [x1]
; ELF-NEXT:    str x8, [x2]
; ELF-NEXT:    ret

  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 8
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  store i64 %tmp1, ptr %dst2
  ret ptr %tmp3
}

; Writeback multiple-use of the auth, invalid offset.

define ptr @test_da_256_wb_use(ptr %ptr, ptr %dst, ptr %dst2) {
; DARWIN-LABEL: test_da_256_wb_use:
; DARWIN:       %bb.0:
; DARWIN-NEXT:    mov x16, x0
; DARWIN-NEXT:    autdza x16
; DARWIN-NEXT:    ldraa x8, [x0, #256]!
; DARWIN-NEXT:    str x8, [x1]
; DARWIN-NEXT:    str x16, [x2]
; DARWIN-NEXT:    ret

; ELF-LABEL: test_da_256_wb_use:
; ELF:       %bb.0:
; ELF-NEXT:    mov x8, x0
; ELF-NEXT:    autdza x8
; ELF-NEXT:    ldraa x9, [x0, #256]!
; ELF-NEXT:    str x9, [x1]
; ELF-NEXT:    str x8, [x2]
; ELF-NEXT:    ret

  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 0)
  %tmp2 = add i64 %tmp1, 256
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  store i64 %tmp4, ptr %dst
  store i64 %tmp1, ptr %dst2
  ret ptr %tmp3
}

; Integer discriminator, no offset.

define i64 @test_da_constdisc(ptr %ptr) {
; CHECK-LABEL: test_da_constdisc:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, #12345
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    ldr x0, [x16]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 12345)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  ret i64 %tmp3
}

define i64 @test_ib_constdisc(ptr %ptr) {
; CHECK-LABEL: test_ib_constdisc:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, #12345
; CHECK-NEXT:    autib x16, x17
; CHECK-NEXT:    ldr x0, [x16]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 1, i64 12345)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  ret i64 %tmp3
}

; "Address" (register) discriminator, no offset.

define i64 @test_da_addrdisc(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_addrdisc:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    autda x16, x1
; CHECK-NEXT:    ldr x0, [x16]
; CHECK-NEXT:    ret
  %tmp0 = ptrtoint ptr %ptr to i64
  %tmp1 = call i64 @llvm.ptrauth.auth(i64 %tmp0, i32 2, i64 %disc)
  %tmp2 = inttoptr i64 %tmp1 to ptr
  %tmp3 = load i64, ptr %tmp2
  ret i64 %tmp3
}

; Blend discriminator, no offset.

define i64 @test_da_blend(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_blend:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    ldr x0, [x16]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = inttoptr i64 %tmp2 to ptr
  %tmp4 = load i64, ptr %tmp3
  ret i64 %tmp4
}

; Blend discriminator, interesting offsets.

define i64 @test_da_blend_8(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_blend_8:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    ldr x0, [x16, #8]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, 8
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  ret i64 %tmp5
}

define i64 @test_da_blend_uimm12max(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_blend_uimm12max:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    ldr x0, [x16, #32760]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, 32760
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  ret i64 %tmp5
}

define i64 @test_da_blend_largeoff_32b(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_blend_largeoff_32b:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    mov x17, #2
; CHECK-NEXT:    movk x17, #1, lsl #32
; CHECK-NEXT:    ldr x0, [x16, x17]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, 4294967298
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  ret i64 %tmp5
}

define i64 @test_da_blend_m4(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_blend_m4:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    ldur x0, [x16, #-4]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, -4
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  ret i64 %tmp5
}

define i64 @test_da_blend_simm9min(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_blend_simm9min:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    mov x17, #-4096
; CHECK-NEXT:    ldr x0, [x16, x17]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, -4096
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  ret i64 %tmp5
}

define i64 @test_da_blend_neg_largeoff_32b(ptr %ptr, i64 %disc) {
; CHECK-LABEL: test_da_blend_neg_largeoff_32b:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    mov x17, #-3
; CHECK-NEXT:    movk x17, #65534, lsl #32
; CHECK-NEXT:    ldr x0, [x16, x17]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, -4294967299
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  ret i64 %tmp5
}

; Blend discriminator, interesting offsets, writeback.

define i64 @test_da_blend_8_wb(ptr %ptr, i64 %disc, ptr %dst) {
; CHECK-LABEL: test_da_blend_8_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    ldr x8, [x16, #8]!
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x2]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, 8
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  store i64 %tmp5, ptr %dst
  ret i64 %tmp3
}

define i64 @test_da_blend_simm9umax_wb(ptr %ptr, i64 %disc, ptr %dst) {
; CHECK-LABEL: test_da_blend_simm9umax_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    ldr x8, [x16, #248]!
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x2]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, 248
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  store i64 %tmp5, ptr %dst
  ret i64 %tmp3
}

define i64 @test_da_blend_simm9s8max_wb(ptr %ptr, i64 %disc, ptr %dst) {
; CHECK-LABEL: test_da_blend_simm9s8max_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    mov x17, #4088
; CHECK-NEXT:    add x16, x16, x17
; CHECK-NEXT:    ldr x8, [x16]
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x2]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, 4088
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  store i64 %tmp5, ptr %dst
  ret i64 %tmp3
}

define i64 @test_da_blend_neg_largeoff_32b_wb(ptr %ptr, i64 %disc, ptr %dst) {
; CHECK-LABEL: test_da_blend_neg_largeoff_32b_wb:
; CHECK:       %bb.0:
; CHECK-NEXT:    mov x16, x0
; CHECK-NEXT:    mov x17, x1
; CHECK-NEXT:    movk x17, #12345, lsl #48
; CHECK-NEXT:    autda x16, x17
; CHECK-NEXT:    mov x17, #-3
; CHECK-NEXT:    movk x17, #65534, lsl #32
; CHECK-NEXT:    add x16, x16, x17
; CHECK-NEXT:    ldr x8, [x16]
; CHECK-NEXT:    mov x0, x16
; CHECK-NEXT:    str x8, [x2]
; CHECK-NEXT:    ret
  %tmp0 = call i64 @llvm.ptrauth.blend(i64 %disc, i64 12345)
  %tmp1 = ptrtoint ptr %ptr to i64
  %tmp2 = call i64 @llvm.ptrauth.auth(i64 %tmp1, i32 2, i64 %tmp0)
  %tmp3 = add i64 %tmp2, -4294967299
  %tmp4 = inttoptr i64 %tmp3 to ptr
  %tmp5 = load i64, ptr %tmp4
  store i64 %tmp5, ptr %dst
  ret i64 %tmp3
}

declare i64 @llvm.ptrauth.auth(i64, i32, i64)
declare i64 @llvm.ptrauth.blend(i64, i64)
