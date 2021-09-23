; Test ST instructions.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 | FileCheck %s

@mem32 = common global i32 0, align 4
@mem16 = common global i16 0, align 2
@mem8 = common global i8 0, align 1

define void @set_mem32(i32 %val) {
; CHECK-LABEL: set_mem32:
; CHECK: or.u %r3, %r0, %hi16(mem32)
; CHECK-NEXT: st %r2, %r3, %lo16(mem32)
; CHECK-NEXT: jmp %r1
  store i32 %val, i32* @mem32, align 4
  ret void
}

define void @set_mem16(i16 %val) {
; CHECK-LABEL: set_mem16:
; CHECK: or.u %r3, %r0, %hi16(mem16)
; CHECK-NEXT: st.h %r2, %r3, %lo16(mem16)
; CHECK-NEXT: jmp %r1
  store i16 %val, i16* @mem16, align 2
  ret void
}

define void @set_mem8(i8 %val) {
; CHECK-LABEL: set_mem8:
; CHECK: or.u %r3, %r0, %hi16(mem8)
; CHECK-NEXT: st.b %r2, %r3, %lo16(mem8)
; CHECK-NEXT: jmp %r1
  store i8 %val, i8* @mem8, align 1
  ret void
}

; CHECK: .comm   mem32,4,4
; CHECK: .comm   mem16,2,2
; CHECK: .comm   mem8,1,1