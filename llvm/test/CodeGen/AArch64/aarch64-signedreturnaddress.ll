; RUN: llc < %s -mtriple=arm64-eabi -asm-verbose=false -mattr=v8.2a | FileCheck %s
; RUN: llc < %s -mtriple=arm64-eabi -asm-verbose=false -mattr=v8.3a | FileCheck %s --check-prefix=CHECKV83

; Armv8.3-A Pointer Authetication requires a special instruction to strip the
; pointer authentication code from the pointer.
; The XPACLRI instruction assembles to a hint-space instruction before Armv8.3-A
; therefore this instruction can be safely used for any pre Armv8.3-A architectures.
; On Armv8.3-A and onwards XPACI is available so use that instead.

define i8* @ra0() nounwind readnone {
entry:
; CHECK-LABEL: ra0:
; CHECK-NEXT:     str     x30, [sp, #-16]!
; CHECK-NEXT:     hint    #7
; CHECK-NEXT:     mov     x0, x30
; CHECK-NEXT:     ldr     x30, [sp], #16
; CHECK-NEXT:     ret
; CHECKV83:       str     x30, [sp, #-16]!
; CHECKV83-NEXT:  xpaci   x30
; CHECKV83-NEXT:  mov     x0, x30
; CHECKV83-NEXT:  ldr     x30, [sp], #16
; CHECKV83-NEXT:  ret
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

define i8* @ra1() nounwind readnone #0 {
entry:
; CHECK-LABEL: ra1:
; CHECK:          hint    #25
; CHECK-NEXT:     str     x30, [sp, #-16]!
; CHECK-NEXT:     hint    #7
; CHECK-NEXT:     mov     x0, x30
; CHECK-NEXT:     ldr     x30, [sp], #16
; CHECK-NEXT:     hint    #29
; CHECK-NEXT:     ret
; CHECKV83:       pacia   x30, sp
; CHECKV83-NEXT:  str     x30, [sp, #-16]!
; CHECKV83-NEXT:  xpaci   x30
; CHECKV83-NEXT:  mov     x0, x30
; CHECKV83-NEXT:  ldr     x30, [sp], #16
; CHECKV83-NEXT:  retaa
  %0 = tail call i8* @llvm.returnaddress(i32 0)
  ret i8* %0
}

attributes #0 = { "sign-return-address"="all" }

declare i8* @llvm.returnaddress(i32) nounwind readnone
