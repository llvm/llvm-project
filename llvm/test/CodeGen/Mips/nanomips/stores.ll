; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

; ----- Tests for byte stores -----

define void @sb_1(i8* %p) {
; CHECK: sb $a1, -256($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -256
  store i8 5, i8* %i, align 1
  ret void
}

define void @sb_2(i8* %p) {
; CHECK-NOT: sb $a1, -257($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -257
  store i8 5, i8* %i, align 1
  ret void
}

define void @sb_3(i8* %p) {
; CHECK: sb $a1, 4095($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4095
  store i8 5, i8* %i, align 1
  ret void
}

define void @sb_4(i8* %p) {
; CHECK-NOT: sb $a1, 4096($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4096
  store i8 5, i8* %i, align 1
  ret void
}

define void @sbx(i8* %p, i32 %n) {
; CHECK: sbx $a2, $a1($a0)
  %nn = sext i32 %n to i64
  %i = getelementptr inbounds i8, i8* %p, i64 %nn
  store i8 5, i8* %i, align 1
  ret void
}

; ----- Tests for short stores -----

define void @sh_1(i16* %p) {
; CHECK: sh $a1, -256($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -128
  store i16 5, i16* %i, align 2
  ret void
}

define void @sh_2(i16* %p) {
; CHECK-NOT: sh $a1, -258($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -129
  store i16 5, i16* %i, align 2
  ret void
}

define void @sh_3(i16* %p) {
; CHECK: sh $a1, 4094($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2047
  store i16 5, i16* %i, align 2
  ret void
}

define void @sh_4(i16* %p) {
; CHECK-NOT: sh $a1, 4096($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2048
  store i16 5, i16* %i, align 2
  ret void
}

define void @shx(i16* %p, i32 %n) {
; CHECK: shx $a2, $a1($a0)
  %nn = sext i32 %n to i64
  %pp = bitcast i16* %p to i8*
  %i = getelementptr inbounds i8, i8* %pp, i64 %nn
  %ii = bitcast i8* %i to i16*
  store i16 5, i16* %ii, align 2
  ret void
}

define void @shxs(i16* %p, i32 %n) {
; CHECK: shxs $a2, $a1($a0)
  %nn = sext i32 %n to i64
  %i = getelementptr inbounds i16, i16* %p, i64 %nn
  store i16 5, i16* %i, align 2
  ret void
}

; ----- Tests for int stores -----

define void @sw_1(i32* %p) {
; CHECK: sw $a1, -256($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 -64
  store i32 5, i32* %i, align 4
  ret void
}

define void @sw_2(i32* %p) {
; CHECK-NOT: sw $a1, -260($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 -65
  store i32 5, i32* %i, align 4
  ret void
}

define void @sw_3(i32* %p) {
; CHECK: sw $a1, 4092($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 1023
  store i32 5, i32* %i, align 4
  ret void
}

define void @sw_4(i32* %p) {
; CHECK-NOT: sw $a1, 4096($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 1024
  store i32 5, i32* %i, align 4
  ret void
}

define void @swx(i32* %p, i32 %n) {
; CHECK: swx $a2, $a1($a0)
  %nn = sext i32 %n to i64
  %pp = bitcast i32* %p to i8*
  %i = getelementptr inbounds i8, i8* %pp, i64 %nn
  %ii = bitcast i8* %i to i32*
  store i32 5, i32* %ii, align 4
  ret void
}

define void @swxs(i32* %p, i32 %n) {
; CHECK: swxs $a2, $a1($a0)
  %nn = sext i32 %n to i64
  %i = getelementptr inbounds i32, i32* %p, i64 %nn
  store i32 5, i32* %i, align 4
  ret void
}
