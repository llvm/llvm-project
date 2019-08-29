; RUN: llc -mtriple=mips64-linux-gnuabi64 \
; RUN:     -relocation-model=pic < %s | FileCheck %s -check-prefix=PIC
; RUN: llc -mtriple=mips64-linux-gnuabi64 \
; RUN:     -relocation-model=static < %s | FileCheck %s -check-prefix=STATIC

define void @bar1() nounwind {
entry:
; PIC:      lui  $[[R0:[0-9]+]], 4095
; PIC-NEXT: ori  $[[R0]], $[[R0]], 65535
; PIC-NEXT: ld   $[[R1:[0-9]+]], %got_disp(foo)(${{[0-9]+}})
; PIC-NEXT: and  $[[R1]], $[[R1]], $[[R0]]
; PIC-NEXT: sd   $[[R1]]

; STATIC:      lui     $[[R0:[0-9]+]], 4095
; STATIC-NEXT: ori     $[[R0]], $[[R0]], 65535
; STATIC-NEXT: daddiu  $[[R1:[0-9]+]], $zero, %hi(foo)
; STATIC-NEXT: dsll    $[[R1]], $[[R1]], 16
; STATIC-NEXT: daddiu  $[[R1]], $[[R1]], %lo(foo)
; STATIC-NEXT: and     $[[R0]], $[[R1]], $[[R0]]
; STATIC-NEXT: sd      $[[R0]]

  %val = alloca i64, align 8
  store i64 and (i64 ptrtoint (void ()* @foo to i64), i64 268435455), i64* %val, align 8
  %0 = load i64, i64* %val, align 8
  ret void
}

declare void @foo()
