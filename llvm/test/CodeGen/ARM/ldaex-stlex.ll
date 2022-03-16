; RUN: llc < %s -mtriple=armv8-apple-darwin   | FileCheck %s
; RUN: llc < %s -mtriple=thumbv8-apple-darwin | FileCheck %s

%0 = type { i32, i32 }

; CHECK-LABEL: f0:
; CHECK: ldaexd
define i64 @f0(i8* %p) nounwind readonly {
entry:
  %ldaexd = tail call %0 @llvm.arm.ldaexd(i8* %p)
  %0 = extractvalue %0 %ldaexd, 1
  %1 = extractvalue %0 %ldaexd, 0
  %2 = zext i32 %0 to i64
  %3 = zext i32 %1 to i64
  %shl = shl nuw i64 %2, 32
  %4 = or i64 %shl, %3
  ret i64 %4
}

; CHECK-LABEL: f1:
; CHECK: stlexd
define i32 @f1(i8* %ptr, i64 %val) nounwind {
entry:
  %tmp4 = trunc i64 %val to i32
  %tmp6 = lshr i64 %val, 32
  %tmp7 = trunc i64 %tmp6 to i32
  %stlexd = tail call i32 @llvm.arm.stlexd(i32 %tmp4, i32 %tmp7, i8* %ptr)
  ret i32 %stlexd
}

declare %0 @llvm.arm.ldaexd(i8*) nounwind readonly
declare i32 @llvm.arm.stlexd(i32, i32, i8*) nounwind

; CHECK-LABEL: test_load_i8:
; CHECK: ldaexb r0, [r0]
; CHECK-NOT: uxtb
; CHECK-NOT: and
define zeroext i8 @test_load_i8(i8* %addr) {
  %val = call i32 @llvm.arm.ldaex.p0i8(i8* elementtype(i8) %addr)
  %val8 = trunc i32 %val to i8
  ret i8 %val8
}

; CHECK-LABEL: test_load_i16:
; CHECK: ldaexh r0, [r0]
; CHECK-NOT: uxth
; CHECK-NOT: and
define zeroext i16 @test_load_i16(i16* %addr) {
  %val = call i32 @llvm.arm.ldaex.p0i16(i16* elementtype(i16) %addr)
  %val16 = trunc i32 %val to i16
  ret i16 %val16
}

; CHECK-LABEL: test_load_i32:
; CHECK: ldaex r0, [r0]
define i32 @test_load_i32(i32* %addr) {
  %val = call i32 @llvm.arm.ldaex.p0i32(i32* elementtype(i32) %addr)
  ret i32 %val
}

declare i32 @llvm.arm.ldaex.p0i8(i8*) nounwind readonly
declare i32 @llvm.arm.ldaex.p0i16(i16*) nounwind readonly
declare i32 @llvm.arm.ldaex.p0i32(i32*) nounwind readonly

; CHECK-LABEL: test_store_i8:
; CHECK-NOT: uxtb
; CHECK: stlexb r0, r1, [r2]
define i32 @test_store_i8(i32, i8 %val, i8* %addr) {
  %extval = zext i8 %val to i32
  %res = call i32 @llvm.arm.stlex.p0i8(i32 %extval, i8* elementtype(i8) %addr)
  ret i32 %res
}

; CHECK-LABEL: test_store_i16:
; CHECK-NOT: uxth
; CHECK: stlexh r0, r1, [r2]
define i32 @test_store_i16(i32, i16 %val, i16* %addr) {
  %extval = zext i16 %val to i32
  %res = call i32 @llvm.arm.stlex.p0i16(i32 %extval, i16* elementtype(i16) %addr)
  ret i32 %res
}

; CHECK-LABEL: test_store_i32:
; CHECK: stlex r0, r1, [r2]
define i32 @test_store_i32(i32, i32 %val, i32* %addr) {
  %res = call i32 @llvm.arm.stlex.p0i32(i32 %val, i32* elementtype(i32) %addr)
  ret i32 %res
}

declare i32 @llvm.arm.stlex.p0i8(i32, i8*) nounwind
declare i32 @llvm.arm.stlex.p0i16(i32, i16*) nounwind
declare i32 @llvm.arm.stlex.p0i32(i32, i32*) nounwind
