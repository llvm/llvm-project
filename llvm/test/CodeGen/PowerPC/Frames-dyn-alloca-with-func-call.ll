; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefix=PPC32-LINUX

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-unknown-linux-gnu \
; RUN: -frame-pointer=all | FileCheck %s -check-prefix=PPC32-LINUX

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu | \
; RUN: FileCheck %s -check-prefixes=PPC64,PPC64-LINUX

; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-unknown-linux-gnu \
; RUN: -frame-pointer=all | FileCheck %s -check-prefixes=PPC64,PPC64-LINUX

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc-ibm-aix-xcoff | FileCheck %s \
; RUN: -check-prefix=PPC32-AIX

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefix=PPC32-AIX

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff | FileCheck %s \
; RUN: -check-prefixes=PPC64,PPC64-AIX

; RUN: llc -verify-machineinstrs < %s -mcpu=pwr4 -mattr=-altivec \
; RUN: -mtriple=powerpc64-ibm-aix-xcoff -frame-pointer=all | FileCheck %s \
; RUN: -check-prefixes=PPC64,PPC64-AIX

define dso_local signext i32 @foo(i32 %n) {
entry:
  %ptr0 = alloca ptr
  %0 = alloca i32, i32 %n
  store ptr %0, ptr %ptr0
  %1 = alloca i32, i32 %n
  %2 = alloca i32, i32 %n
  %3 = alloca i32, i32 %n
  %4 = alloca i32, i32 %n
  %5 = alloca i32, i32 %n
  %6 = alloca i32, i32 %n
  %7 = alloca i32, i32 %n
  %8 = alloca i32, i32 %n
  %9 = load ptr, ptr %ptr0

  %call = call i32 @bar(ptr %1, ptr %2, ptr %3, ptr %4, ptr %5, ptr %6, ptr %7, ptr %8, ptr %9)
  ret i32 %call
}

declare i32 @bar(ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr)

; PPC32-LINUX-LABEL: foo
; PPC32-LINUX: mflr 0
; PPC32-LINUX: stw 0, 4(1)
; PPC32-LINUX: stwu 1, -32(1)
; PPC32-LINUX: stw 31, 28(1)
; PPC32-LINUX: mr 31, 1
; PPC32-LINUX: addi 3, 31, 32
; PPC32-LINUX: stwux 3, 1, 10

; Allocated area is referred by stack pointer.
; PPC32-LINUX: addi 11, 1, 16

; Local variable area is referred by frame pointer.
; PPC32-LINUX: stw 11, 24(31)

; PPC32-LINUX: stwux
; PPC32-LINUX: stwux
; PPC32-LINUX: stwux
; PPC32-LINUX: stwux
; PPC32-LINUX: stwux
; PPC32-LINUX: stwux
; PPC32-LINUX: stwux
; PPC32-LINUX: stwux
; PPC32-LINUX-NOT: stuwux

; Parameter save area is referred by stack pointer.
; PPC32-LINUX: stw 11, 8(1)

; PPC32-LINUX: bl bar
; PPC32-LINUX: lwz 31, 0(1)
; PPC32-LINUX: lwz 0, -4(31)
; PPC32-LINUX: mr 1, 31
; PPC32-LINUX: mr 31, 0
; PPC32-LINUX: lwz 0, 4(1)
; PPC32-LINUX: mtlr 0
; PPC32-LINUX: blr

; PPC64-LABEL: foo
; PPC64: mflr 0
; PPC64: std 31, -8(1)
; PPC64: std 0, 16(1)
; PPC64: stdu 1, -160(1)
; PPC64: mr 31, 1
; PPC64: addi 3, 31, 160
; PPC64: stdux 3, 1, 10

; Allocated area is referred by stack pointer.
; PPC64: addi 11, 1, 128

; Local variable area is referred by frame pointer.
; PPC64: std 11, 144(31)

; PPC64: stdux
; PPC64: stdux
; PPC64: stdux
; PPC64: stdux
; PPC64: stdux
; PPC64: stdux
; PPC64: stdux
; PPC64: stdux
; PPC64-NOT: stdux

; Parameter save area is referred by stack pointer.
; PPC64: std 11, 112(1)

; PPC64-LINUX: bl bar
; PPC64-AIX: bl .bar
; PPC64: ld 1, 0(1)
; PPC64: ld 0, 16(1)
; PPC64-DAG: ld 31, -8(1)
; PPC64-DAG: mtlr 0
; PPC64: blr

; PPC32-AIX: mflr 0
; PPC32-AIX: stw 31, -4(1)
; PPC32-AIX: stw 0, 8(1)
; PPC32-AIX: stwu 1, -80(1)
; PPC32-AIX: mr 31, 1
; PPC32-AIX: addi 3, 31, 80
; PPC32-AIX: stwux 3, 1, 10

; Allocated area is referred by stack pointer.
; PPC32-AIX: addi 11, 1, 64

; Local variable area is referred by frame pointer.
; PPC32-AIX: stw 11, 72(31)

; PPC32-AIX: stwux
; PPC32-AIX: stwux
; PPC32-AIX: stwux
; PPC32-AIX: stwux
; PPC32-AIX: stwux
; PPC32-AIX: stwux
; PPC32-AIX: stwux
; PPC32-AIX: stwux
; PPC32-AIX-NOT: stwux

; Parameter save area is referred by stack pointer.
; PPC32-AIX: stw 11, 56(1)

; PPC32-AIX: bl .bar
; PPC32-AIX: nop
; PPC32-AIX: lwz 1, 0(1)
; PPC32-AIX: lwz 0, 8(1)
; PPC32-AIX-DAG: mtlr 0
; PPC32-AIX-DAG: lwz 31, -4(1)
; PPC32-AIX: blr
