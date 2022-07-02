; Test generation of constants.
;
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88100 -m88k-enable-delay-slot-filler=false | FileCheck %s
; RUN: llc < %s -mtriple=m88k-openbsd -mcpu=mc88110 -m88k-enable-delay-slot-filler=false | FileCheck %s

define zeroext i16 @f1() {
; CHECK-LABEL: f1:
; CHECK: or %r2, %r0, 0
; CHECK: jmp %r1
  ret i16 0
}

define i32 @f2() {
; CHECK-LABEL: f2:
; CHECK: or %r2, %r0, 0
; CHECK: jmp %r1
  ret i32 0
}

define zeroext i16 @f3() {
; CHECK-LABEL: f3:
; CHECK: or %r2, %r0, 1
; CHECK: jmp %r1
  ret i16 1
}

define i32 @f4() {
; CHECK-LABEL: f4:
; CHECK: or %r2, %r0, 1
; CHECK: jmp %r1
  ret i32 1
}

define i32 @f5() {
; CHECK-LABEL: f5:
; CHECK: or.u %r2, %r0, 1
; CHECK: jmp %r1
  ret i32 65536 ; 0x10000
}

define i32 @f6() {
; CHECK-LABEL: f6:
; CHECK: subu %r2, %r0, 1
; CHECK: jmp %r1
  ret i32 -1
}

define i32 @f7() {
; CHECK-LABEL: f7:
; CHECK: subu %r2, %r0, 65535
; CHECK: jmp %r1
  ret i32 -65535
}

define i32 @f8() {
; CHECK-LABEL: f8:
; CHECK: set %r2, %r0, 16<16>
; CHECK: jmp %r1
  ret i32 -65536
}

define zeroext i16 @f9() {
; CHECK-LABEL: f9:
; CHECK: or %r2, %r0, 51966
; CHECK: jmp %r1
  ret i16 51966 ; 0xcafe
}

define i32 @f10() {
; CHECK-LABEL: f10:
; 51966 = 0xcafe, 47806 = 0xbabe
; CHECK: or.u %r2, %r0, 51966
; CHECK: or %r2, %r2, 47806
; CHECK: jmp %r1
  ret i32 3405691582 ; 0xcafebabe
}

define i64 @f11() {
; CHECK-LABEL: f11:
; CHECK: or %r3, %r0, 0
; CHECK: or %r2, %r0, 0
; CHECK: jmp %r1
  ret i64 0
}

define i64 @f12() {
; CHECK-LABEL: f12:
; CHECK: or %r3, %r0, 1
; CHECK: or %r2, %r0, 0
; CHECK: jmp %r1
  ret i64 1
}

define i64 @f13() {
; CHECK-LABEL: f13:
; 51966 = 0xcafe, 47806 = 0xbabe
; CHECK: or.u %r2, %r0, 51966
; CHECK: or   %r3, %r2, 47806
; CHECK: or   %r2, %r0, 0
; CHECK: jmp %r1
  ret i64 3405691582 ; 0xcafebabe
}

define i64 @f14() {
; CHECK-LABEL: f14:
; 57005 = 0xdead, 48879 = 0xbeef
; 51966 = 0xcafe, 47806 = 0xbabe
; CHECK: or.u %r2, %r0, 51966
; CHECK: or   %r3, %r2, 47806
; CHECK: or.u %r2, %r0, 57005
; CHECK: or   %r2, %r2, 48879
; CHECK: jmp %r1
  ret i64 -2401053089206453570 ; 0xcafebabe
}
