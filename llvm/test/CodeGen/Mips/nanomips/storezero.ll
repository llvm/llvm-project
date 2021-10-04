; RUN: llc -march=nanomips -asm-show-inst -verify-machineinstrs -mload-store-unaligned < %s | FileCheck %s
; Tests store zero peephole optimization for all store instructions.

define void @zeroint0(i32* %num) {
; CHECK: sw $zero, 0($a0)
  store i32 0, i32* %num, align 4
  ret void
}

define void @zeroint1(i32* %num) {
  %arrayidx = getelementptr inbounds i32, i32* %num, i32 1
; CHECK: sw $zero, 4($a0)
  store i32 0, i32* %arrayidx, align 4
  ret void
}

define void @zeroint2(i32* %num) {
  %arrayidx = getelementptr inbounds i32, i32* %num, i32 -1
; CHECK: sw $zero, -4($a0)
  store i32 0, i32* %arrayidx, align 4
  ret void
}

define void @zeroint3(i32* %num) {
  %arrayidx = getelementptr inbounds i32, i32* %num, i32 10000
; CHECK: swx $zero, $a1($a0)
  store i32 0, i32* %arrayidx, align 4
  ret void
}

define void @zeroshort0(i16* %num) {
; CHECK: sh $zero, 0($a0)
  store i16 0, i16* %num, align 2
  ret void
}

define void @zeroshort1(i16* %num) {
  %arrayidx = getelementptr inbounds i16, i16* %num, i32 1
; CHECK: sh $zero, 2($a0)
  store i16 0, i16* %arrayidx, align 2
  ret void
}

define void @zeroshort2(i16* %num) {
  %arrayidx = getelementptr inbounds i16, i16* %num, i32 -1
; CHECK: sh $zero, -2($a0)
  store i16 0, i16* %arrayidx, align 2
  ret void
}

define void @zeroshort3(i16* %num) {
  %arrayidx = getelementptr inbounds i16, i16* %num, i32 10000
; CHECK: shx $zero, $a1($a0)
  store i16 0, i16* %arrayidx, align 2
  ret void
}

define void @zerochar0(i8* %num) {
; CHECK: sb $zero, 0($a0)
  store i8 0, i8* %num, align 1
  ret void
}

define void @zerochar1(i8* %num) {
  %arrayidx = getelementptr inbounds i8, i8* %num, i32 1
; CHECK: sb $zero, 1($a0)
  store i8 0, i8* %arrayidx, align 1
  ret void
}

define void @zerochar2(i8* %num) {
  %arrayidx = getelementptr inbounds i8, i8* %num, i32 -1
; CHECK: sb $zero, -1($a0)
  store i8 0, i8* %arrayidx, align 1
  ret void
}

define void @zerochar3(i8* %num) {
  %arrayidx = getelementptr inbounds i8, i8* %num, i32 10000
; CHECK: sbx $zero, $a1($a0)
  store i8 0, i8* %arrayidx, align 1
  ret void
}

define void @zerointunaligned0(i32* %num) {
; CHECK: uasw $zero, 0($a0)
  store i32 0, i32* %num, align 2
  ret void
}

define void @zerointunaligned1(i32* %num) {
  %arrayidx = getelementptr inbounds i32, i32* %num, i32 1
; CHECK: uasw $zero, 4($a0)
  store i32 0, i32* %arrayidx, align 2
  ret void
}

define void @zerointunaligned2(i32* %num) {
  %arrayidx = getelementptr inbounds i32, i32* %num, i32 -1
; CHECK: uasw $zero, -4($a0)
  store i32 0, i32* %arrayidx, align 2
  ret void
}

define void @zerointunaligned3(i32* %num) {
  %arrayidx = getelementptr inbounds i32, i32* %num, i32 10000
; CHECK: uasw $zero, 0($a0)
  store i32 0, i32* %arrayidx, align 2
  ret void
}

; TODO: Select uash for unaligned halfword stores.
define void @zeroshortunaligned0(i16* %num) {
; CHECK-NOT: uash $zero, 0($a0)
  store i16 0, i16* %num, align 1
  ret void
}

; TODO: Select uash for unaligned halfword stores.
define void @zeroshortunaligned1(i16* %num) {
  %arrayidx = getelementptr inbounds i16, i16* %num, i32 1
; CHECK-NOT: uash $zero, 2($a0)
  store i16 0, i16* %arrayidx, align 1
  ret void
}

; TODO: Select uash for unaligned halfword stores.
define void @zeroshortunaligned2(i16* %num) {
  %arrayidx = getelementptr inbounds i16, i16* %num, i32 -1
; CHECK-NOT: uash $zero, -2($a0)
  store i16 0, i16* %arrayidx, align 1
  ret void
}

; TODO: Select uash for unaligned halfword stores.
define void @zeroshortunaligned3(i16* %num) {
  %arrayidx = getelementptr inbounds i16, i16* %num, i32 10000
; CHECK-NOT: uash $zero, 0($a0)
  store i16 0, i16* %arrayidx, align 1
  ret void
}
