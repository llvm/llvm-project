; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

; equivalent C code
;   struct s64 {
;   	int a:5;
;   	int b:16;
;   	long c:42;
;   };
;   void bitfieldinsert64(struct s *p, unsigned short v) {
;   	p->b = v;
;   }

%struct.s64 = type { i64 }

define void @bitfieldinsert64(ptr nocapture %p, i16 zeroext %v) {
; CHECK-LABEL: @bitfieldinsert64
; CHECK: ld [[REG1:[0-9]+]], 0(3)
; CHECK-NEXT: rlwimi [[REG1]], 4, 5, 11, 26
; CHECK-NEXT: std [[REG1]], 0(3)
; CHECK-NEXT: blr
entry:
  %0 = zext i16 %v to i64
  %bf.load = load i64, ptr %p, align 8
  %bf.shl = shl nuw nsw i64 %0, 5
  %bf.clear = and i64 %bf.load, -2097121
  %bf.set = or i64 %bf.clear, %bf.shl
  store i64 %bf.set, ptr %p, align 8
  ret void
}

; bitfieldinsert32: Test for rlwimi
; equivalent C code
;   struct s32 {
;   	int a:8;
;   	int b:16;
;   	int c:8;
;   };
;   void bitfieldinsert32(struct s32 *p, unsigned int v) {
;   	p->b = v;
;   }

%struct.s32 = type { i32 }

define void @bitfieldinsert32(ptr nocapture %p, i32 zeroext %v) {
; CHECK-LABEL: @bitfieldinsert32
; CHECK: lwz [[REG1:[0-9]+]], 0(3)
; CHECK-NEXT: rlwimi [[REG1]], 4, 8, 8, 23
; CHECK-NEXT: stw [[REG1]], 0(3)
; CHECK-NEXT: blr
entry:
  %bf.load = load i32, ptr %p, align 4
  %bf.value = shl i32 %v, 8
  %bf.shl = and i32 %bf.value, 16776960
  %bf.clear = and i32 %bf.load, -16776961
  %bf.set = or i32 %bf.clear, %bf.shl
  store i32 %bf.set, ptr %p, align 4
  ret void
}

; test cases which include ISD::TRUNCATE
; equivalent C code
;   struct s64b {
;     int a:4;
;     int b:16;
;     int c:24;
;   };
;   void bitfieldinsert64b(struct s64b *p, unsigned char v) {
;     p->b = v;
;   }

%struct.s64b = type { i24, i24 }

define void @bitfieldinsert64b(ptr nocapture %p, i8 zeroext %v) {
; CHECK-LABEL: @bitfieldinsert64b
; CHECK: lwz [[REG1:[0-9]+]], 0(3)
; CHECK-NEXT: rlwimi [[REG1]], 4, 4, 12, 27
; CHECK-NEXT: stw [[REG1]], 0(3)
; CHECK-NEXT: blr
entry:
  %conv = zext i8 %v to i32
  %bf.load = load i32, ptr %p, align 4
  %bf.shl = shl nuw nsw i32 %conv, 4
  %bf.clear = and i32 %bf.load, -1048561
  %bf.set = or i32 %bf.clear, %bf.shl
  store i32 %bf.set, ptr %p, align 4
  ret void
}

; equivalent C code
;   struct s64c {
;     int a:5;
;     int b:16;
;     long c:10;
;   };
;   void bitfieldinsert64c(struct s64c *p, unsigned short v) {
;     p->b = v;
;   }

%struct.s64c = type { i32, [4 x i8] }

define void @bitfieldinsert64c(ptr nocapture %p, i16 zeroext %v) {
; CHECK-LABEL: @bitfieldinsert64c
; CHECK: lwz [[REG1:[0-9]+]], 0(3)
; CHECK-NEXT: rlwimi [[REG1]], 4, 5, 11, 26
; CHECK-NEXT: stw [[REG1]], 0(3)
; CHECK-NEXT: blr
entry:
  %conv = zext i16 %v to i32
  %bf.load = load i32, ptr %p, align 8
  %bf.shl = shl nuw nsw i32 %conv, 5
  %bf.clear = and i32 %bf.load, -2097121
  %bf.set = or i32 %bf.clear, %bf.shl
  store i32 %bf.set, ptr %p, align 8
  ret void
}
