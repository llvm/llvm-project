;; -ffunction-sections previously resulted in call relocations against the
;; .text.foo section symbol instead of the actual function symbol.
;; However, that results in a relocation against a symbol without the LSB set,
;; so the linker thinks it is not linking against a Thumb symbol.
;; Check that we use a relocation against the symbol for calls to functions
;; marked as dso_local (foo$local)
;; NB: Right now only R_ARM_PREL31 and R_ARM_ABS32 are converted to section
;; plus offset, so this test can't use a call.
; RUN: llc -mtriple=armv7a-none-linux-gnueabi --function-sections -o - --relocation-model=pic %s | FileCheck %s
; RUN: llc -mtriple=armv7a-none-linux-gnueabi --function-sections --filetype=obj -o %t --relocation-model=pic %s
; RUN: llvm-readobj -r --symbols %t | FileCheck %s --check-prefix=RELOCS
;; Do not autogen (we check directives that are normally filtered out):
; UTC-ARGS: --disable

; RELOCS-LABEL: Relocations [
; RELOCS-NEXT:   Section (5) .rel.ARM.exidx.text._ZdlPv {
; RELOCS-NEXT:     0x0 R_ARM_PREL31 .text._ZdlPv
; RELOCS-NEXT:   }
; RELOCS-NEXT:   Section (7) .rel.text.test {
; RELOCS-NEXT:     0x4 R_ARM_CALL .L_ZdlPv$local
; RELOCS-NEXT:     0xC R_ARM_ABS32 .L_ZdlPv$local
; RELOCS-NEXT:     0x10 R_ARM_ABS32 .L_ZdlPv$local
; RELOCS-NEXT:     0x1C R_ARM_REL32 .L_ZdlPv$local
; RELOCS-NEXT:   }
; RELOCS-NEXT:   Section (9) .rel.ARM.exidx.text.test {
; RELOCS-NEXT:     0x0 R_ARM_PREL31 .text.test
; RELOCS-NEXT:   }
; RELOCS-NEXT:   Section (11) .rel.data {
; RELOCS-NEXT:     0x0 R_ARM_ABS32 _ZdlPv
; RELOCS-NEXT:   }
; RELOCS-NEXT: ]

; RELOCS-LABEL: Symbols [
; RELOCS:      Symbol {
; RELOCS:        Name: .L_ZdlPv$local
; RELOCS-NEXT:   Value: 0x1
; RELOCS-NEXT:   Size: 2
; RELOCS-NEXT:   Binding: Local (0x0)
; RELOCS-NEXT:   Type: Function (0x2)
; RELOCS-NEXT:   Other: 0
; RELOCS-NEXT:   Section: .text._ZdlPv (
; RELOCS-NEXT: }

define dso_local void @_ZdlPv(ptr %ptr) local_unnamed_addr nounwind "target-features"="+armv7-a,+thumb-mode" {
; CHECK-LABEL: 	.section	.text._ZdlPv,"ax",%progbits
; CHECK-NEXT: 	.globl	_ZdlPv                          @ -- Begin function _ZdlPv
; CHECK-NEXT: 	.p2align	1
; CHECK-NEXT: 	.type	_ZdlPv,%function
; CHECK-NEXT: 	.code	16                              @ @_ZdlPv
; CHECK-NEXT: 	.thumb_func
; CHECK-NEXT: _ZdlPv:
; CHECK-NEXT: .L_ZdlPv$local:
; CHECK-NEXT: .type .L_ZdlPv$local,%function
; CHECK-NEXT: 	.fnstart
; CHECK-NEXT: @ %bb.0:
; CHECK-NEXT: 	bx	lr
; CHECK-NEXT: .Lfunc_end0:
; CHECK-NEXT: 	.size	_ZdlPv, .Lfunc_end0-_ZdlPv
; CHECK-NEXT: 	.size .L_ZdlPv$local, .Lfunc_end0-_ZdlPv
; CHECK-NEXT: 	.cantunwind
; CHECK-NEXT: 	.fnend
  ret void
}

define ptr @test(ptr %ptr) nounwind {
; CHECK-LABEL: test:
; CHECK:       @ %bb.0: @ %entry
; CHECK-NEXT:    .save {r11, lr}
; CHECK-NEXT:    push {r11, lr}
; CHECK-NEXT:    bl .L_ZdlPv$local
; CHECK-NEXT:    ldr r0, .LCPI1_0
; CHECK-NEXT:    @APP
; CHECK-NEXT:    .long .L_ZdlPv$local
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:    @APP
; CHECK-NEXT:    .Ltmp0:
; CHECK-NEXT:    .reloc .Ltmp0, R_ARM_ABS32, .L_ZdlPv$local
; CHECK-NEXT:    .long 0
; CHECK-NEXT:    @NO_APP
; CHECK-NEXT:  .LPC1_0:
; CHECK-NEXT:    add r0, pc, r0
; CHECK-NEXT:    pop {r11, pc}
; CHECK-NEXT:    .p2align 2
; CHECK-NEXT:  @ %bb.1:
; CHECK-NEXT:  .LCPI1_0:
; CHECK-NEXT:    .long .L_ZdlPv$local-(.LPC1_0+8)
entry:
  call void @_ZdlPv(ptr %ptr)
  ; This inline assembly is needed to highlight the missing Thumb LSB since
  ; only R_ARM_ABS32 is converted to section+offset
  tail call void asm sideeffect ".4byte .L_ZdlPv$$local", ""()
  tail call void asm sideeffect ".reloc ., R_ARM_ABS32, .L_ZdlPv$$local\0A.4byte 0", ""()
  ret ptr @_ZdlPv
}

@fnptr = hidden local_unnamed_addr global ptr @_ZdlPv
