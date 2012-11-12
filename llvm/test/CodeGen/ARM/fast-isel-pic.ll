; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=pic -mtriple=thumbv7-apple-ios | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=pic -mtriple=arm-apple-ios | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=pic -mtriple=armv7-apple-ios | FileCheck %s --check-prefix=ARMv7
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=pic -mtriple=thumbv7-none-linux-gnueabi | FileCheck %s --check-prefix=THUMB-ELF
; RUN: llc < %s -O0 -fast-isel-abort -relocation-model=pic -mtriple=armv7-none-linux-gnueabi | FileCheck %s --check-prefix=ARMv7-ELF

@g = global i32 0, align 4

define i32 @LoadGV() {
entry:
; THUMB: LoadGV
; THUMB: movw [[reg0:r[0-9]+]],
; THUMB: movt [[reg0]],
; THUMB: add  [[reg0]], pc
; THUMB-ELF: LoadGV
; THUMB-ELF: ldr.n r[[reg0:[0-9]+]],
; THUMB-ELF: ldr.n r[[reg1:[0-9]+]],
; THUMB-ELF: ldr r[[reg0]], [r[[reg1]], r[[reg0]]]
; ARM: LoadGV
; ARM: ldr [[reg1:r[0-9]+]],
; ARM: add [[reg1]], pc, [[reg1]]
; ARMv7: LoadGV
; ARMv7: movw [[reg2:r[0-9]+]],
; ARMv7: movt [[reg2]],
; ARMv7: add  [[reg2]], pc, [[reg2]]
; ARMv7-ELF: LoadGV
; ARMv7-ELF: ldr r[[reg2:[0-9]+]],
; ARMv7-ELF: ldr r[[reg3:[0-9]+]],
; ARMv7-ELF: ldr r[[reg2]], [r[[reg3]], r[[reg2]]]
  %tmp = load i32* @g
  ret i32 %tmp
}

@i = external global i32

define i32 @LoadIndirectSymbol() {
entry:
; THUMB: LoadIndirectSymbol
; THUMB: movw r[[reg3:[0-9]+]],
; THUMB: movt r[[reg3]],
; THUMB: add  r[[reg3]], pc
; THUMB: ldr  r[[reg3]], [r[[reg3]]]
; THUMB-ELF: LoadIndirectSymbol
; THUMB-ELF: ldr.n r[[reg3:[0-9]+]],
; THUMB-ELF: ldr.n r[[reg4:[0-9]+]],
; THUMB-ELF: ldr r[[reg3]], [r[[reg4]], r[[reg3]]]
; ARM: LoadIndirectSymbol
; ARM: ldr [[reg4:r[0-9]+]],
; ARM: ldr [[reg4]], [pc, [[reg4]]]
; ARMv7: LoadIndirectSymbol
; ARMv7: movw r[[reg5:[0-9]+]],
; ARMv7: movt r[[reg5]],
; ARMv7: add  r[[reg5]], pc, r[[reg5]]
; ARMv7: ldr  r[[reg5]], [r[[reg5]]]
; ARMv7-ELF: LoadIndirectSymbol
; ARMv7-ELF: ldr r[[reg5:[0-9]+]],
; ARMv7-ELF: ldr r[[reg6:[0-9]+]],
; ARMv7-ELF: ldr r[[reg5]], [r[[reg6]], r[[reg5]]]
  %tmp = load i32* @i
  ret i32 %tmp
}
