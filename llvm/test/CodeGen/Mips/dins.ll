; RUN: llc -O2 -verify-machineinstrs -march=mips64 -mcpu=mips64r2 \
; RUN:   -target-abi=n64 < %s -o - | FileCheck %s -check-prefix=MIPS64R2
; RUN: llc -O2 -verify-machineinstrs -march=mips -mcpu=mips32r2 < %s -o - \
; RUN:   | FileCheck %s -check-prefix=MIPS32R2
; RUN: llc -O2 -verify-machineinstrs -march=mips -mattr=mips16 < %s -o - \
; RUN:   | FileCheck %s -check-prefix=MIPS16
; RUN: llc -O2 -verify-machineinstrs -march=mips64 -mcpu=mips64r2 \
; RUN:   -target-abi=n32 < %s -o - | FileCheck %s -check-prefix=MIPS64R2N32

; #include <stdint.h>
; #include <stdio.h>
; struct cvmx_buf_ptr {

;   struct {
;     unsigned long long addr :37;
;     unsigned long long addr1 :15;
;     unsigned int lenght:14;
;     uint64_t total_bytes:16;
;     uint64_t segs : 6;
;   } s;
; }
;
; unsigned long long foo(volatile struct cvmx_buf_ptr bufptr) {
;   bufptr.s.addr = 123;
;   bufptr.s.segs = 4;
;   bufptr.s.lenght = 5;
;   bufptr.s.total_bytes = bufptr.s.lenght;
;   return bufptr.s.addr;
; }

; Testing of selection INS/DINS instruction

define i64 @f123(i64 inreg %bufptr.coerce0, i64 inreg %bufptr.coerce1) local_unnamed_addr #0 {
entry:
  %bufptr.sroa.0 = alloca i64, align 8
  %bufptr.sroa.4 = alloca i64, align 8
  store i64 %bufptr.coerce0, i64* %bufptr.sroa.0, align 8
  store i64 %bufptr.coerce1, i64* %bufptr.sroa.4, align 8
  %bufptr.sroa.0.0.bufptr.sroa.0.0.bufptr.sroa.0.0.bf.load = load volatile i64, i64* %bufptr.sroa.0, align 8
  %bf.clear = and i64 %bufptr.sroa.0.0.bufptr.sroa.0.0.bufptr.sroa.0.0.bf.load, 134217727
  %bf.set = or i64 %bf.clear, 16508780544
  store volatile i64 %bf.set, i64* %bufptr.sroa.0, align 8
  %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load2 = load volatile i64, i64* %bufptr.sroa.4, align 8
  %bf.clear3 = and i64 %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load2, -16911433729
  %bf.set4 = or i64 %bf.clear3, 1073741824
  store volatile i64 %bf.set4, i64* %bufptr.sroa.4, align 8
  %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load6 = load volatile i64, i64* %bufptr.sroa.4, align 8
  %bf.clear7 = and i64 %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load6, 1125899906842623
  %bf.set8 = or i64 %bf.clear7, 5629499534213120
  store volatile i64 %bf.set8, i64* %bufptr.sroa.4, align 8
  %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load11 = load volatile i64, i64* %bufptr.sroa.4, align 8
  %bf.lshr = lshr i64 %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load11, 50
  %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load13 = load volatile i64, i64* %bufptr.sroa.4, align 8
  %bf.shl = shl nuw nsw i64 %bf.lshr, 34
  %bf.clear14 = and i64 %bufptr.sroa.4.0.bufptr.sroa.4.0.bufptr.sroa.4.8.bf.load13, -1125882726973441
  %bf.set15 = or i64 %bf.clear14, %bf.shl
  store volatile i64 %bf.set15, i64* %bufptr.sroa.4, align 8
  %bufptr.sroa.0.0.bufptr.sroa.0.0.bufptr.sroa.0.0.bf.load17 = load volatile i64, i64* %bufptr.sroa.0, align 8
  %bf.lshr18 = lshr i64 %bufptr.sroa.0.0.bufptr.sroa.0.0.bufptr.sroa.0.0.bf.load17, 27
  ret i64 %bf.lshr18
}

; CHECK-LABEL: f123:
; MIPS64R2: daddiu  $[[R0:[0-9]+]], $zero, 123
; MIPS64R2: dinsm   $[[R0:[0-9]+]], $[[R1:[0-9]+]], 27, 37
; MIPS64R2: daddiu  $[[R0:[0-9]+]], $zero, 4
; MIPS64R2: dinsm   $[[R0:[0-9]+]], $[[R1:[0-9]+]], 28, 6
; MIPS64R2: daddiu  $[[R0:[0-9]+]], $zero, 5
; MIPS64R2: dinsu   $[[R0:[0-9]+]], $[[R1:[0-9]+]], 50, 14
; MIPS64R2: dsrl    $[[R0:[0-9]+]], $[[R1:[0-9]+]], 50
; MIPS64R2: dinsu   $[[R0:[0-9]+]], $[[R1:[0-9]+]], 34, 16
; MIPS32R2: ins     $[[R0:[0-9]+]], $[[R1:[0-9]+]], 2, 16
; MIPS32R2-NOT: ins $[[R0:[0-9]+]], $[[R1:[0-9]+]], 18, 46
; MIPS16-NOT: ins{{[[:space:]].*}}


; int foo(volatile int x) {
;   int y = x;
;   y = y & -4;
;   x = y | 8;
;   return y;
; }

define i32 @foo(i32 signext %x) {
entry:
  %x.addr = alloca i32, align 4
  store volatile i32 %x, i32* %x.addr, align 4
  %x.addr.0.x.addr.0. = load volatile i32, i32* %x.addr, align 4
  %and = and i32 %x.addr.0.x.addr.0., -4
  %or = or i32 %and, 8
  store volatile i32 %or, i32* %x.addr, align 4
  ret i32 %and
}

; CHECK-LABEL: foo:
; MIPS64R2:        ori  $[[R0:[0-9]+]], $[[R0:[0-9]+]], 8
; MIPS64R2-NOT:    ins {{[[:space:]].*}}
; MIPS32R2:        ori  $[[R0:[0-9]+]], $[[R0:[0-9]+]], 8
; MIPS32R2-NOT:    ins {{[[:space:]].*}}
; MIPS64R2N32:     ori  $[[R0:[0-9]+]], $[[R0:[0-9]+]], 8
; MIPS64R2N32-NOT: ins {{[[:space:]].*}}
