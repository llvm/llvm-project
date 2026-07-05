; RUN: llc < %s -mtriple=armv7-apple-darwin   | FileCheck %s
; RUN: llc < %s -mtriple=thumbv7-apple-darwin > %t
; RUN: FileCheck %s < %t
; RUN: FileCheck %s < %t --check-prefix=CHECK-T2ADDRMODE

%0 = type { i32, i32 }

; CHECK-LABEL: f0:
; CHECK: ldrexd
define i64 @f0(ptr %p) nounwind readonly {
entry:
  %ldrexd = tail call %0 @llvm.arm.ldrexd(ptr %p)
  %0 = extractvalue %0 %ldrexd, 1
  %1 = extractvalue %0 %ldrexd, 0
  %2 = zext i32 %0 to i64
  %3 = zext i32 %1 to i64
  %shl = shl nuw i64 %2, 32
  %4 = or i64 %shl, %3
  ret i64 %4
}

; CHECK-LABEL: f1:
; CHECK: strexd
define i32 @f1(ptr %ptr, i64 %val) nounwind {
entry:
  %tmp4 = trunc i64 %val to i32
  %tmp6 = lshr i64 %val, 32
  %tmp7 = trunc i64 %tmp6 to i32
  %strexd = tail call i32 @llvm.arm.strexd(i32 %tmp4, i32 %tmp7, ptr %ptr)
  ret i32 %strexd
}

declare %0 @llvm.arm.ldrexd(ptr) nounwind readonly
declare i32 @llvm.arm.strexd(i32, i32, ptr) nounwind

; CHECK-LABEL: test_load_i8:
; CHECK: ldrexb r0, [r0]
; CHECK-NOT: uxtb
; CHECK-NOT: and
define zeroext i8 @test_load_i8(ptr %addr) {
  %val = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i8) %addr)
  %val8 = trunc i32 %val to i8
  ret i8 %val8
}

; CHECK-LABEL: test_load_i16:
; CHECK: ldrexh r0, [r0]
; CHECK-NOT: uxth
; CHECK-NOT: and
define zeroext i16 @test_load_i16(ptr %addr) {
  %val = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i16) %addr)
  %val16 = trunc i32 %val to i16
  ret i16 %val16
}

; CHECK-LABEL: test_load_i32:
; CHECK: ldrex r0, [r0]
define i32 @test_load_i32(ptr %addr) {
  %val = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %addr)
  ret i32 %val
}

declare i32 @llvm.arm.ldrex.p0(ptr) nounwind readonly

; CHECK-LABEL: test_store_i8:
; CHECK-NOT: uxtb
; CHECK: strexb r0, r1, [r2]
define i32 @test_store_i8(i32, i8 %val, ptr %addr) {
  %extval = zext i8 %val to i32
  %res = call i32 @llvm.arm.strex.p0(i32 %extval, ptr elementtype(i8) %addr)
  ret i32 %res
}

; CHECK-LABEL: test_store_i16:
; CHECK-NOT: uxth
; CHECK: strexh r0, r1, [r2]
define i32 @test_store_i16(i32, i16 %val, ptr %addr) {
  %extval = zext i16 %val to i32
  %res = call i32 @llvm.arm.strex.p0(i32 %extval, ptr elementtype(i16) %addr)
  ret i32 %res
}

; CHECK-LABEL: test_store_i32:
; CHECK: strex r0, r1, [r2]
define i32 @test_store_i32(i32, i32 %val, ptr %addr) {
  %res = call i32 @llvm.arm.strex.p0(i32 %val, ptr elementtype(i32) %addr)
  ret i32 %res
}

declare i32 @llvm.arm.strex.p0(i32, ptr) nounwind

; CHECK-LABEL: test_clear:
; CHECK: clrex
define void @test_clear() {
  call void @llvm.arm.clrex()
  ret void
}

declare void @llvm.arm.clrex() nounwind

@base = global ptr null

define void @excl_addrmode() {
; CHECK-T2ADDRMODE-LABEL: excl_addrmode:
  %base1020 = load ptr, ptr @base
  %offset1020 = getelementptr i32, ptr %base1020, i32 255
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %offset1020)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %offset1020)
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [{{r[0-9]+}}, #1020]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [{{r[0-9]+}}, #1020]

  %base1024 = load ptr, ptr @base
  %offset1024 = getelementptr i32, ptr %base1024, i32 256
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %offset1024)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %offset1024)
; CHECK-T2ADDRMODE: add.w r[[ADDR:[0-9]+]], {{r[0-9]+}}, #1024
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]

  %base1 = load ptr, ptr @base
  %offset1_8 = getelementptr i8, ptr %base1, i32 1
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %offset1_8)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %offset1_8)
; CHECK-T2ADDRMODE: adds r[[ADDR:[0-9]+]], #1
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]

  %local = alloca i8, i32 1024
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %local)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %local)
; CHECK-T2ADDRMODE: mov r[[ADDR:[0-9]+]], sp
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]

  ret void
}

define void @test_excl_addrmode_folded() {
; CHECK-LABEL: test_excl_addrmode_folded:
  %local = alloca i8, i32 4096

  %local.0 = getelementptr i8, ptr %local, i32 4
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %local.0)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %local.0)
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [sp, #4]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [sp, #4]

  %local.1 = getelementptr i8, ptr %local, i32 1020
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %local.1)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %local.1)
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [sp, #1020]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [sp, #1020]

  ret void
}

define void @test_excl_addrmode_range() {
; CHECK-LABEL: test_excl_addrmode_range:
  %local = alloca i8, i32 4096

  %local.0 = getelementptr i8, ptr %local, i32 1024
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %local.0)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %local.0)
; CHECK-T2ADDRMODE: mov r[[TMP:[0-9]+]], sp
; CHECK-T2ADDRMODE: add.w r[[ADDR:[0-9]+]], r[[TMP]], #1024
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]

  ret void
}

define void @test_excl_addrmode_align() {
; CHECK-LABEL: test_excl_addrmode_align:
  %local = alloca i8, i32 4096

  %local.0 = getelementptr i8, ptr %local, i32 2
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %local.0)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %local.0)
; CHECK-T2ADDRMODE: mov r[[ADDR:[0-9]+]], sp
; CHECK-T2ADDRMODE: adds r[[ADDR:[0-9]+]], #2
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]

  ret void
}

define void @test_excl_addrmode_sign() {
; CHECK-LABEL: test_excl_addrmode_sign:
  %local = alloca i8, i32 4096

  %local.0 = getelementptr i8, ptr %local, i32 -4
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %local.0)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %local.0)
; CHECK-T2ADDRMODE: mov r[[ADDR:[0-9]+]], sp
; CHECK-T2ADDRMODE: subs r[[ADDR:[0-9]+]], #4
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [r[[ADDR]]]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [r[[ADDR]]]

  ret void
}

define void @test_excl_addrmode_combination() {
; CHECK-LABEL: test_excl_addrmode_combination:
  %local = alloca i8, i32 4096
  %unused = alloca i8, i32 64

  %local.0 = getelementptr i8, ptr %local, i32 4
  call i32 @llvm.arm.ldrex.p0(ptr elementtype(i32) %local.0)
  call i32 @llvm.arm.strex.p0(i32 0, ptr elementtype(i32) %local.0)
; CHECK-T2ADDRMODE: ldrex {{r[0-9]+}}, [sp, #68]
; CHECK-T2ADDRMODE: strex {{r[0-9]+}}, {{r[0-9]+}}, [sp, #68]

  ret void
}


; LLVM should know, even across basic blocks, that ldrex is setting the high
; bits of its i32 to 0. There should be no zero-extend operation.
define zeroext i8 @test_cross_block_zext_i8(i1 %tst, ptr %addr) {
; CHECK: test_cross_block_zext_i8:
; CHECK-NOT: uxtb
; CHECK-NOT: and
; CHECK: bx lr
  %val = call i32 @llvm.arm.ldrex.p0(ptr elementtype(i8) %addr)
  br i1 %tst, label %end, label %mid
mid:
  ret i8 42
end:
  %val8 = trunc i32 %val to i8
  ret i8 %val8
}
