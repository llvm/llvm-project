; RUN: llc -O2 -mtriple=riscv32 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=ZBB
; RUN: llc -O2 -mtriple=riscv32 -mattr=-zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s | FileCheck %s --check-prefix=NOZBB
; RUN: llc -O0 -mtriple=riscv32 -mattr=+zbb -global-isel -global-isel-abort=1 -verify-machineinstrs < %s -o /dev/null

define i32 @byte_bit_0(i32 %x) {
; ZBB-LABEL: byte_bit_0:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_0:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 16843009
  %l = shl i32 %m, 8
  %r = sub i32 %l, %m
  ret i32 %r
}

define i32 @byte_bit_1(i32 %x) {
; ZBB-LABEL: byte_bit_1:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_1:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 33686018
  %l = shl i32 %m, 7
  %h = lshr i32 %m, 1
  %r = sub i32 %l, %h
  ret i32 %r
}

define i32 @byte_bit_2(i32 %x) {
; ZBB-LABEL: byte_bit_2:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_2:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 67372036
  %l = shl i32 %m, 6
  %h = lshr i32 %m, 2
  %r = sub i32 %l, %h
  ret i32 %r
}

define i32 @byte_bit_3(i32 %x) {
; ZBB-LABEL: byte_bit_3:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_3:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 134744072
  %l = shl i32 %m, 5
  %h = lshr i32 %m, 3
  %r = sub i32 %l, %h
  ret i32 %r
}

define i32 @byte_bit_4(i32 %x) {
; ZBB-LABEL: byte_bit_4:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_4:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 269488144
  %l = shl i32 %m, 4
  %h = lshr i32 %m, 4
  %r = sub i32 %l, %h
  ret i32 %r
}

define i32 @byte_bit_5(i32 %x) {
; ZBB-LABEL: byte_bit_5:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_5:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 538976288
  %l = shl i32 %m, 3
  %h = lshr i32 %m, 5
  %r = sub i32 %l, %h
  ret i32 %r
}

define i32 @byte_bit_6(i32 %x) {
; ZBB-LABEL: byte_bit_6:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_6:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, 1077952576
  %l = shl i32 %m, 2
  %h = lshr i32 %m, 6
  %r = sub i32 %l, %h
  ret i32 %r
}

define i32 @byte_bit_7(i32 %x) {
; ZBB-LABEL: byte_bit_7:
; ZBB: orc.b
; ZBB: ret
; NOZBB-LABEL: byte_bit_7:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, -2139062144
  %l = shl i32 %m, 1
  %h = lshr i32 %m, 7
  %r = sub i32 %l, %h
  ret i32 %r
}

define i32 @no_known_bits(i32 %x) {
; ZBB-LABEL: no_known_bits:
; ZBB-NOT: orc.b
; ZBB: ret
; NOZBB-LABEL: no_known_bits:
; NOZBB-NOT: orc.b
; NOZBB: ret
  %m = and i32 %x, -1
  %l = shl i32 %m, 8
  %r = sub i32 %l, %m
  ret i32 %r
}
