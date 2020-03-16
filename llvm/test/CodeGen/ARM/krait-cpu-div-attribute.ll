; Tests the genration of ".arch_extension" attribute for hardware
; division on krait CPU. For now, krait is recognized as "cortex-a9" + hwdiv
; Also, tests for the hwdiv instruction on krait CPU

; check for arch_extension/cpu directive
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=krait | FileCheck %s --check-prefix=DIV_EXTENSION
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -mcpu=krait | FileCheck %s --check-prefix=DIV_EXTENSION
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=cortex-a9 | FileCheck %s --check-prefix=NODIV_KRAIT
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -mcpu=cortex-a9 | FileCheck %s --check-prefix=NODIV_KRAIT
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=krait -mattr=-hwdiv,-hwdiv-arm | FileCheck %s --check-prefix=NODIV_KRAIT

; check if correct instruction is emitted by integrated assembler
; RUN: llc < %s -mtriple=armv7-linux-gnueabi -mcpu=krait -filetype=obj | llvm-objdump --mcpu=krait --triple=armv7-linux-gnueabi -d - | FileCheck %s --check-prefix=HWDIV
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -mcpu=krait -filetype=obj | llvm-objdump --mcpu=krait --triple=thumbv7-linux-gnueabi -d - | FileCheck %s --check-prefix=HWDIV

; arch_extension attribute
; DIV_EXTENSION:  .cpu cortex-a9
; DIV_EXTENSION:  .arch_extension idiv
; NODIV_KRAIT-NOT:  .arch_extension idiv
; HWDIV: sdiv

define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 0, i32* %retval
  store volatile i32 100, i32* %b, align 4
  store volatile i32 32, i32* %c, align 4
  %0 = load volatile i32, i32* %b, align 4
  %1 = load volatile i32, i32* %c, align 4
  %div = sdiv i32 %0, %1
  store volatile i32 %div, i32* %a, align 4
  ret i32 0
}
