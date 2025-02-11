; RUN: llc -mtriple=mips < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN: llc -mtriple=mipsel < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN-TODO: llc -mtriple=mips64 -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s
; RUN-TODO: llc -mtriple=mips64el -target-abi o32 < %s | FileCheck --check-prefixes=ALL,O32 %s

; RUN: llc -mtriple=mips64 -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s
; RUN: llc -mtriple=mips64el -target-abi n32 < %s | FileCheck --check-prefixes=ALL,N32 %s

; RUN: llc -mtriple=mips64 -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s
; RUN: llc -mtriple=mips64el -target-abi n64 < %s | FileCheck --check-prefixes=ALL,N64 %s

; Test that O32 correctly reserved space for the four arguments, even when
; there aren't any as per section 5 of MD00305 (MIPS ABIs Described).

declare void @foo() nounwind;

define void @reserved_space() nounwind {
entry:
        call void @foo()
        ret void
}

; ALL-LABEL: reserved_space:
; O32:           addiu $sp, $sp, -24
; O32:           sw $ra, 20($sp)
; O32:           lw $ra, 20($sp)
; O32:           addiu $sp, $sp, 24
; Despite pointers being 32-bit wide on N32, the return pointer is saved as a
; 64-bit pointer. I've yet to find a documentation reference for this quirk but
; this behaviour matches GCC so I have considered it to be correct.
; N32:           addiu $sp, $sp, -16
; N32:           sd $ra, 8($sp)
; N32:           ld $ra, 8($sp)
; N32:           addiu $sp, $sp, 16
; N64:           daddiu $sp, $sp, -16
; N64:           sd $ra, 8($sp)
; N64:           ld $ra, 8($sp)
; N64:           daddiu $sp, $sp, 16
