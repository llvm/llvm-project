; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

; Ranges are tested only with LB, other patterns are tested only to make sure they are actually selected.

define signext i8 @lb_0(i8* %p) {
; CHECK-LABEL: lb_0
; CHECK-NOT: addiu $a0, $a0, -256
; CHECK-NOT: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -256
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_1(i8* %p) {
; CHECK-LABEL: lb_1
; CHECK: addiu $a0, $a0, -257
; CHECK: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -257
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_2(i8* %p) {
; CHECK-LABEL: lb_2
; CHECK: addiu $a0, $a0, -4095
; CHECK: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -4095
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_3(i8* %p) {
; CHECK-LABEL: lb_3
; CHECK-NOT: addiu $a0, $a0, -4096
; CHECK-NOT: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -4096
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_4(i8* %p) {
; CHECK-LABEL: lb_4
; CHECK-NOT: addiu $a0, $a0, 4095
; CHECK-NOT: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4095
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_5(i8* %p) {
; CHECK-LABEL: lb_5
; CHECK: addiu $a0, $a0, 4096
; CHECK: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4096
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_6(i8* %p) {
; CHECK-LABEL: lb_6
; CHECK: addiu $a0, $a0, 65535
; CHECK: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 65535
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_7(i8* %p) {
; CHECK-LABEL: lb_7
; CHECK-NOT: addiu $a0, $a0, 65536
; CHECK-NOT: lb $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 65536
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define zeroext i8 @test_lbu(i8* %p) {
; CHECK-LABEL: test_lbu
; CHECK: addiu $a0, $a0, -257
; CHECK: lbu $a0, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -257
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i16 @test_lh(i16* %p) {
; CHECK-LABEL: test_lh
; CHECK: addiu $a0, $a0, -258
; CHECK: lh $a0, 0($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -129
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define zeroext i16 @test_lhu(i16* %p) {
; CHECK-LABEL: test_lhu
; CHECK: addiu $a0, $a0, -258
; CHECK: lhu $a0, 0($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -129
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define i32 @test_lw(i32* %p) {
; CHECK-LABEL: test_lw
; CHECK: addiu $a0, $a0, -260
; CHECK: lw $a0, 0($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 -65
  %v = load i32, i32* %i, align 4
  ret i32 %v
}

define void @test_sb(i8* %p) {
; CHECK-LABEL: test_sb
; CHECK: addiu $a0, $a0, -257
; CHECK: sb $a1, 0($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -257
  store i8 5, i8* %i, align 1
  ret void
}

define void @test_sh(i16* %p) {
; CHECK-LABEL: test_sh
; CHECK: addiu $a0, $a0, -258
; CHECK: sh $a1, 0($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -129
  store i16 5, i16* %i, align 2
  ret void
}

define void @test_sw(i32* %p) {
; CHECK-LABEL: test_sw
; CHECK: addiu $a0, $a0, -260
; CHECK: sw $a1, 0($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 -65
  store i32 5, i32* %i, align 4
  ret void
}
