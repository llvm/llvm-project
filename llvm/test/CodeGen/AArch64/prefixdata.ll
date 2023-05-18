; RUN: llc < %s -mtriple=aarch64-apple-darwin | FileCheck --check-prefix=MACHO %s
; RUN: llc < %s -mtriple=aarch64-pc-linux | FileCheck --check-prefix=ELF %s

@i = linkonce_odr global i32 1

; MACHO: ltmp0:
; MACHO-NEXT: .long 1
; MACHO-NEXT: .alt_entry _f
; MACHO-NEXT: _f:
; ELF: .type f,@function
; ELF-NEXT: .word	1  // 0x1
; ELF-NEXT: f:  // @f
define void @f() prefix i32 1 {
  ret void
}

; MACHO: ltmp1:
; MACHO-NEXT: .quad _i
; MACHO-NEXT: .alt_entry _g
; MACHO-NEXT: _g:
; ELF: .type g,@function
; ELF-NEXT: .xword	i
; ELF-NEXT: g:
define void @g() prefix ptr @i {
  ret void
}

; MACHO: .subsections_via_symbols
