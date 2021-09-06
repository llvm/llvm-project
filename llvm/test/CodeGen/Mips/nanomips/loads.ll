; RUN: llc -mtriple=nanomips -asm-show-inst -verify-machineinstrs < %s | FileCheck %s

define signext i8 @lb_1(i8* %p) {
; CHECK: lb $a0, -256($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -256
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_2(i8* %p) {
; CHECK-NOT: lb $a0, -257($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -257
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_3(i8* %p) {
; CHECK: lb $a0, 4095($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4095
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i8 @lb_4(i8* %p) {
; CHECK-NOT: lb $a0, 4096($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4096
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define i8 @lbu_1(i8* %p) {
; CHECK: lbu $a0, -256($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -256
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define i8 @lbu_2(i8* %p) {
; CHECK-NOT: lbu $a0, -257($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -257
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define i8 @lbu_3(i8* %p) {
; CHECK: lbu $a0, 4095($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4095
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define i8 @lbu_4(i8* %p) {
; CHECK-NOT: lbu $a0, 4096($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4096
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define zeroext i8 @lbu_5(i8* %p) {
; CHECK: lbu $a0, -256($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -256
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define zeroext i8 @lbu_6(i8* %p) {
; CHECK-NOT: lbu $a0, -257($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 -257
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define zeroext i8 @lbu_7(i8* %p) {
; CHECK: lbu $a0, 4095($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4095
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define zeroext i8 @lbu_8(i8* %p) {
; CHECK-NOT: lbu $a0, 4096($a0)
  %i = getelementptr inbounds i8, i8* %p, i64 4096
  %v = load i8, i8* %i, align 1
  ret i8 %v
}

define signext i16 @lh_1(i16* %p) {
; CHECK: lh $a0, -256($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -128
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define signext i16 @lh_2(i16* %p) {
; CHECK-NOT: lh $a0, -258($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -129
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define signext i16 @lh_3(i16* %p) {
; CHECK: lh $a0, 4094($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2047
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define signext i16 @lh_4(i16* %p) {
; CHECK-NOT: lh $a0, 4096($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2048
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define i16 @lhu_1(i16* %p) {
; CHECK: lhu $a0, -256($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -128
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define i16 @lhu_2(i16* %p) {
; CHECK-NOT: lhu $a0, -258($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -129
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define i16 @lhu_3(i16* %p) {
; CHECK: lhu $a0, 4094($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2047
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define i16 @lhu_4(i16* %p) {
; CHECK-NOT: lhu $a0, 4096($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2048
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define zeroext i16 @lhu_5(i16* %p) {
; CHECK: lhu $a0, -256($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -128
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define zeroext i16 @lhu_6(i16* %p) {
; CHECK-NOT: lhu $a0, -258($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 -129
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define zeroext i16 @lhu_7(i16* %p) {
; CHECK: lhu $a0, 4094($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2047
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define zeroext i16 @lhu_8(i16* %p) {
; CHECK-NOT: lhu $a0, 4096($a0)
  %i = getelementptr inbounds i16, i16* %p, i64 2048
  %v = load i16, i16* %i, align 2
  ret i16 %v
}

define i32 @lw_1(i32* %p) {
; CHECK: lw $a0, -256($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 -64
  %v = load i32, i32* %i, align 4
  ret i32 %v
}

define i32 @lw_2(i32* %p) {
; CHECK-NOT: lw $a0, -260($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 -65
  %v = load i32, i32* %i, align 4
  ret i32 %v
}

define i32 @lw_3(i32* %p) {
; CHECK: lw $a0, 4092($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 1023
  %v = load i32, i32* %i, align 4
  ret i32 %v
}

define i32 @lw_4(i32* %p) {
; CHECK-NOT: lw $a0, 4096($a0)
  %i = getelementptr inbounds i32, i32* %p, i64 1024
  %v = load i32, i32* %i, align 4
  ret i32 %v
}
