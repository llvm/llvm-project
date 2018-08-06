; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck --check-prefix=MACHO %s
; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck --check-prefix=ELF %s

;MACHO: .globl _offsetSym0
;MACHO-NOT: .alt_entry
;MACHO: .set _offsetSym0, _s
;MACHO: .globl _offsetSym1
;MACHO: .alt_entry _offsetSym1
;MACHO: .set _offsetSym1, _s+8

;ELF: .globl offsetSym0
;ELF-NOT: .alt_entry
;ELF: .set offsetSym0, s
;ELF: .globl offsetSym1
;ELF-NOT: .alt_entry
;ELF: .set offsetSym1, s+8

%struct.S1 = type { i32, i32, i32 }

@s = global %struct.S1 { i32 31, i32 32, i32 33 }, align 4
@offsetSym0 = alias i32, i32* getelementptr inbounds (%struct.S1, %struct.S1* @s, i64 0, i32 0)
@offsetSym1 = alias i32, i32* getelementptr inbounds (%struct.S1, %struct.S1* @s, i64 0, i32 2)
