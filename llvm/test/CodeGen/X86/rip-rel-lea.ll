; RUN: llc < %s -mtriple=x86_64-pc-linux-gnu -relocation-model=pic | FileCheck %s -check-prefix=PIC64
; RUN: llc < %s -mtriple=x86_64-pc-linux-gnux32 -relocation-model=pic | FileCheck %s -check-prefix=PICX32
; RUN: llc < %s -mtriple=i686-pc-linux-gnu -relocation-model=pic | FileCheck %s -check-prefix=PIC32

; Use %rip-relative addressing even in static mode on x86-64, because
; it has a smaller encoding.

@a = internal global double 3.4
define ptr @foo() nounwind {
  ret ptr @a
  
; PIC64:    leaq	a(%rip)
; PICX32:   leal	a(%rip)
; PIC32:    leal	a@GOTOFF(%eax)
}
