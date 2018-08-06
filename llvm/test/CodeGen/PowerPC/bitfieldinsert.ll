; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck %s

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

define void @bitfieldinsert32(%struct.s32* nocapture %p, i32 zeroext %v) {
; CHECK-LABEL: @bitfieldinsert32
; CHECK: lwz [[REG1:[0-9]+]], 0(3)
; CHECK: rlwimi [[REG1]], 4, 8, 8, 23
; CHECK: stw [[REG1]], 0(3)
; CHECK: blr
entry:
  %0 = getelementptr inbounds %struct.s32, %struct.s32* %p, i64 0, i32 0
  %bf.load = load i32, i32* %0, align 4
  %bf.value = shl i32 %v, 8
  %bf.shl = and i32 %bf.value, 16776960
  %bf.clear = and i32 %bf.load, -16776961
  %bf.set = or i32 %bf.clear, %bf.shl
  store i32 %bf.set, i32* %0, align 4
  ret void
}

