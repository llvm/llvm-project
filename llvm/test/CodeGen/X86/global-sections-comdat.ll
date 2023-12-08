; RUN: llc < %s -mtriple=i386-unknown-linux | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=i386-unknown-linux -data-sections -function-sections | FileCheck %s -check-prefix=LINUX-SECTIONS
; RUN: llc < %s -mtriple=i386-unknown-linux -data-sections -function-sections -unique-section-names=false | FileCheck %s -check-prefix=LINUX-SECTIONS-SHORT

$F1 = comdat any
define void @F1(i32 %y) comdat {
bb0:
switch i32 %y, label %bb5 [
    i32 1, label %bb1
    i32 2, label %bb2
    i32 3, label %bb3
    i32 4, label %bb4
  ]
bb1:
  ret void
bb2:
  ret void
bb3:
  ret void
bb4:
  ret void
bb5:
  ret void
}

; LINUX: .section        .text.F1,"axG",@progbits,F1,comdat
; LINUX: .size   F1,
; LINUX-NEXT: .cfi_endproc
; LINUX-NEXT: .section        .rodata.F1,"aG",@progbits,F1,comdat

; LINUX-SECTIONS: .section        .text.F1,"axG",@progbits,F1,comdat
; LINUX-SECTIONS: .size   F1,
; LINUX-SECTIONS-NEXT: .cfi_endproc
; LINUX-SECTIONS-NEXT: .section        .rodata.F1,"aG",@progbits,F1,comdat

; LINUX-SECTIONS-SHORT: .section        .text,"axG",@progbits,F1,comdat
; LINUX-SECTIONS-SHORT: .size   F1,
; LINUX-SECTIONS-SHORT-NEXT: .cfi_endproc
; LINUX-SECTIONS-SHORT-NEXT: .section        .rodata,"aG",@progbits,F1,comdat

$G16 = comdat any
@G16 = unnamed_addr constant i32 42, comdat

; LINUX: .section	.rodata.cst4.G16,"aGM",@progbits,4,G16,comdat
; LINUX-SECTIONS: .section	.rodata.cst4.G16,"aGM",@progbits,4,G16,comdat
; LINUX-SECTIONS-SHORT: .section	.rodata.cst4,"aGM",@progbits,4,G16,comdat
