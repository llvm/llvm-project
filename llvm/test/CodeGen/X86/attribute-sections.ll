; RUN: llc < %s -mtriple=i386-unknown-linux-gnu | FileCheck %s -check-prefix=LINUX

declare i32 @foo()
@G0 = global i32 ()* @foo, section ".init_array"

; LINUX:  .section  .init_array,"aw"
; LINUX:  .globl G0

@G1 = global i32 ()* @foo, section ".fini_array"

; LINUX:  .section  .fini_array,"aw"
; LINUX:  .globl G1

@G2 = global i32 ()* @foo, section ".preinit_array"

; LINUX:  .section .preinit_array,"aw"
; LINUX:  .globl G2

