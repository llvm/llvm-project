; RUN: llc -march=mipsel -mcpu=mips32 < %s | FileCheck %s
; RUN: llc -march=mips64el -mcpu=mips4 < %s | \
; RUN:      FileCheck %s -check-prefix=CHECK-MIPS64
; RUN: llc -march=mips64el -mcpu=mips64 < %s | \
; RUN:      FileCheck %s -check-prefix=CHECK-MIPS64

declare ptr @llvm.eh.dwarf.cfa(i32) nounwind
declare ptr @llvm.frameaddress(i32) nounwind readnone

define ptr @f1() nounwind {
entry:
  %x = alloca [32 x i8], align 1
  %0 = call ptr @llvm.eh.dwarf.cfa(i32 0)
  ret ptr %0

; CHECK-LABEL: f1:

; CHECK:        addiu   $sp, $sp, -32
; CHECK:        addiu   $2, $sp, 32
}


define ptr @f2() nounwind {
entry:
  %x = alloca [65536 x i8], align 1
  %0 = call ptr @llvm.eh.dwarf.cfa(i32 0)
  ret ptr %0

; CHECK-LABEL: f2:

; check stack size (65536 + 8)
; CHECK:        lui     $[[R0:[a-z0-9]+]], 1
; CHECK:        addiu   $[[R0]], $[[R0]], 8
; CHECK:        subu    $sp, $sp, $[[R0]]

; check return value ($sp + stack size)
; CHECK:        lui     $[[R1:[a-z0-9]+]], 1
; CHECK:        addu    $[[R1]], $sp, $[[R1]]
; CHECK:        addiu   $2, $[[R1]], 8
}


define i32 @f3() nounwind {
entry:
  %x = alloca [32 x i8], align 1
  %0 = call ptr @llvm.eh.dwarf.cfa(i32 0)
  %1 = ptrtoint ptr %0 to i32
  %2 = call ptr @llvm.frameaddress(i32 0)
  %3 = ptrtoint ptr %2 to i32
  %add = add i32 %1, %3
  ret i32 %add

; CHECK-LABEL: f3:

; CHECK:        addiu   $sp, $sp, -40

; check return value ($fp + stack size + $fp)
; CHECK:        addiu   $[[R0:[a-z0-9]+]], $fp, 40
; CHECK:        addu    $2, $[[R0]], $fp
}


define ptr @f4() nounwind {
entry:
  %x = alloca [32 x i8], align 1
  %0 = call ptr @llvm.eh.dwarf.cfa(i32 0)
  ret ptr %0

; CHECK-LABEL: f4:

; CHECK-MIPS64:        daddiu   $sp, $sp, -32
; CHECK-MIPS64:        daddiu   $2, $sp, 32
}
