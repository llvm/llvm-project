; RUN: llc -march=mipsel -mcpu=mips32 < %s | FileCheck %s -check-prefix=CHECK-MIPS32
; RUN: llc -march=mips64el -mcpu=mips64 < %s | \
; RUN:      FileCheck %s -check-prefix=CHECK-MIPS64
; RUN: llc -march=mipsel -mcpu=mips64 -target-abi n32 < %s | \
; RUN:      FileCheck %s -check-prefix=CHECK-MIPSN32

; Test that the expansion of ADJCALLSTACKDOWN and ADJCALLSTACKUP generate
; (d)subu and (d)addu rather than just (d)addu. The (d)subu sequences are
; generally shorter as the constant that has to be materialized is smaller.

define i32 @main() {
entry:
  %z = alloca [1048576 x i8], align 1
  %arraydecay = getelementptr inbounds [1048576 x i8], [1048576 x i8]* %z, i32 0, i32 0
  %call = call i32 @foo(i8* %arraydecay)
  ret i32 0
; CHECK-LABEL: main

; CHECK-MIPS32: lui   $[[R0:[0-9]+]], 16
; CHECK-MIPS32: addiu $[[R0]], $[[R0]], 24
; CHECK-MIPS32: subu  $sp, $sp, $[[R0]]

; CHECK-MIPS32: lui   $[[R1:[0-9]+]], 16
; CHECK-MIPS32: addiu $[[R1]], $[[R1]], 24
; CHECK-MIPS32: addu  $sp, $sp, $[[R1]]

; CHECK-MIPS64: lui     $[[R0:[0-9]+]], 1
; CHECK-MIPS64: daddiu  $[[R0]], $[[R0]], 32
; CHECK-MIPS64: dsubu   $sp, $sp, $[[R0]]

; FIXME:
; These are here to match other lui's used in address computations. We need to
; investigate why address computations are not CSE'd. Or implement it.

; CHECK-MIPS64: lui
; CHECK-MIPS64: lui
; CHECK-MIPS64: lui
; CHECK-MIPS64: lui

; CHECK-MIPS64: lui     $[[R1:[0-9]+]], 16
; CHECK-MIPS64: daddiu  $[[R1]], $[[R1]], 32
; CHECK-MIPS64: daddu   $sp, $sp, $[[R1]]

; CHECK-MIPSN32: lui   $[[R0:[0-9]+]], 16
; CHECK-MIPSN32: addiu $[[R0]], $[[R0]], 16
; CHECK-MIPSN32: subu  $sp, $sp, $[[R0]]

; CHECK-MIPSN32: lui   $[[R1:[0-9]+]], 16
; CHECK-MIPSN32: addiu $[[R1]], $[[R1]], 16
; CHECK-MIPSN32: addu  $sp, $sp, $[[R1]]

}

declare i32 @foo(i8*)
